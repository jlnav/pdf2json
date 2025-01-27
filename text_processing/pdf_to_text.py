from transformers import AutoProcessor, VisionEncoderDecoderModel
import torch
import json
from pathlib import Path
from PIL import Image
import io
import pymupdf as fitz  # https://pymupdf.readthedocs.io/en/latest/installation.html#problems-after-installation
from typing import Optional, List
from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict

# jlnav:
# it appears most of the below is from:
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Nougat/Inference_with_Nougat_to_read_scientific_PDFs.ipynb

# Model and processor loading
processor = AutoProcessor.from_pretrained("facebook/nougat-small")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("mps") if torch.backends.mps.is_available() else device  # for macOS metal backend
model.to(device)


class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0


def rasterize_paper(
    pdf: Path,
    outpath: Optional[Path] = None,
    dpi: int = 96,
    return_pil: bool = False,
    pages: Optional[List[int]] = None
) -> Optional[List[io.BytesIO]]:
    """
    Rasterize a PDF file to PNG images.

    Args:
        pdf (Path): The path to the PDF file to be rasterized.
        outpath (Optional[Path], optional): The output directory for saving images. 
            If None, the PIL images will be returned. Defaults to None.
        dpi (int, optional): The resolution (dots per inch) of the rasterized images. 
            Higher DPI results in better quality but larger file sizes. Defaults to 96.
        return_pil (bool, optional): If True, the function returns a list of PIL image objects 
            as in-memory byte streams (instead of saving them to disk). Defaults to False.
        pages (Optional[List[int]], optional): The list of page numbers to rasterize. 
            If None, all pages are rasterized. Defaults to None.

    Returns:
        Optional[List[io.BytesIO]]: If `return_pil` is True, returns a list of in-memory byte streams 
        containing PIL images for each page. Otherwise, returns None after saving images to the output path.
    """
    
    # List to hold PIL image byte streams, if requested
    pillow_images = []
    
    # If no output path is provided, we return PIL images by default
    if outpath is None:
        return_pil = True

    try:
        # Open the PDF file using PyMuPDF (fitz)
        if isinstance(pdf, (str, Path)):
            pdf = fitz.open(str(pdf))

        # If no specific pages are provided, rasterize all pages
        if pages is None:
            pages = range(len(pdf))

        # Loop over each page
        for i in pages:
            # Get the page's pixmap (image data) and convert it to PNG bytes
            page_pixmap = pdf[i].get_pixmap(dpi=dpi)
            page_bytes = page_pixmap.pil_tobytes(format="PNG")

            # If return_pil is True, store the image as an in-memory byte stream
            if return_pil:
                pillow_images.append(io.BytesIO(page_bytes))
            else:
                # Otherwise, save the image to the specified output directory
                image_path = outpath / f"page_{i + 1:02d}.png"
                with image_path.open("wb") as f:
                    f.write(page_bytes)

    except Exception as e:
        print(f"Error while rasterizing the PDF: {e}")

    # If returning PIL images, return the list of byte streams
    if return_pil:
        return pillow_images    


def pdf2text(filepath: str):

    print('Input file is:' + filepath)

    images = rasterize_paper(pdf=filepath, return_pil=True)
    num_pages = len(images)
    print('Number of pages:' + str(num_pages))

    # Initialize an array to store generated texts
    text_samples = []

    for page_num in range(num_pages):
        print('page num:' + str(page_num))

        image = Image.open(images[page_num])

        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        outputs = model.generate(
            pixel_values.to(device),
            min_length=1,
            max_length=3584,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
        )

        generated = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]

        generated = processor.post_process_generation(generated, fix_markdown=False)
        # print(generated)
        text_samples.append(generated)

    return text_samples



def text2json(text_samples: List[str], fileOut: str):
    # Construct the JSON structure
    data = {
        "type": "text_only",
        "instances": [{"text": sample} for sample in text_samples]
    }

    # Serialize to JSON
    json_data = json.dumps(data, indent=4)

    # Save to file
    
    with open(fileOut + '.json', 'w') as f:
        f.write(json_data)

    print('File saved at: ' + fileOut)
