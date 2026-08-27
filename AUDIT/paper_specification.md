# Paper Specification (VGST1 methodology)

NOTE: The primary paper PDF `docs/vgst1-methodology.pdf` is present in the repository but cannot be reliably parsed by this auditor agent as a binary PDF without an external PDF text-extraction tool. To avoid fabricating or misrepresenting paper values, the following file records the plan and a conservative placeholder: I will not invent or assert any PAPER-SPECIFIED values until the PDF is manually inspected or a text-extraction tool is run.

Planned actions to complete Phase 2:

1. Extract full text from `docs/vgst1-methodology.pdf` using a PDF text extraction tool (pdftotext/poppler or Python `pdfminer.six`).
2. Parse the extracted text for the methodology sections and produce a structured specification covering:
   - dataset definitions and subject splits
   - preprocessing and TR/voxel dims
   - sequence construction (window length, stride)
   - alignment methods and exact formulations
   - Transformer architecture (layers, heads, dims)
   - CLIP/DINO branch architectures and objectives
   - diffusion priors (dimensions, scheduler, loss terms)
   - fusion and retrieval details
   - Stable Diffusion reconstruction hyperparameters
   - training stages (epochs, batch sizes, optimizer)
   - evaluation metrics and statistical tests
   - ablation and data-efficiency experimental details
3. Classify each extracted value as one of: PAPER-SPECIFIED, IMPLEMENTATION-CHOICE, NOT-SPECIFIED.
4. Produce `AUDIT/paper_specification.md` with the above information and citations (paper section / page numbers).

Interim conservative notes (do NOT treat as paper-specified):
- Notebooks reference `vgst1.pdf` for dataset assumptions and flow; I will cross-check notebooks next and mark any explicit values found there as IMPLEMENTATION-CHOICE until confirmed by the paper.

Next step: run a PDF-to-text extractor locally or ask to install `pdfminer.six` or `poppler` tools in this environment so I can extract the text automatically and continue. If you prefer, I will proceed to Phase 3 (notebooks) now and extract workflow details there (notebook code is the secondary source).
