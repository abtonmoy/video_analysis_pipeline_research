from graphviz import Digraph
from IPython.display import Image

dot = Digraph('Layered_Ad_Pipeline')
dot.attr(rankdir='LR')

# ===== LAYER 1 – INGESTION =====
with dot.subgraph(name='cluster_ingestion') as c:
    c.attr(label='INGESTION LAYER', style='rounded')
    c.node('V', 'VideoLoader\nload()')
    c.node('A', 'AudioExtractor\nextract_audio()')

with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('V')
    s.node('A')

# ===== LAYER 2 – SCENE ANALYSIS =====
with dot.subgraph(name='cluster_scenes') as c:
    c.attr(label='SCENE DETECTION LAYER', style='rounded')
    c.node('S2a', 'Primary SceneDetector')
    c.node('S2b', 'Fallback Detector')
    c.node('S2c', 'Artificial Chunking')

dot.edge('S2a', 'S2b')
dot.edge('S2b', 'S2c')

# ===== LAYER 3 – FRAME EXTRACTION =====
with dot.subgraph(name='cluster_candidates') as c:
    c.attr(label='CANDIDATE FRAME LAYER', style='rounded')
    c.node('C3a', 'ChangeDetector')
    c.node('C3b', 'CandidateFrameExtractor')
    c.node('C3c', 'Extracted Frames')

dot.edge('C3a', 'C3b')
dot.edge('C3b', 'C3c')

# ===== LAYER 4 – DEDUPLICATION =====
with dot.subgraph(name='cluster_dedup') as c:
    c.attr(label='DEDUPLICATION LAYER', style='rounded')
    c.node('D4a', 'Hash Voting')
    c.node('D4b', 'CLIP Embeddings')
    c.node('D4c', 'LPIPS Similarity')
    c.node('D4d', 'Remove Near Duplicates')

with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('D4b')
    s.node('D4c')

dot.edge('D4a', 'D4b')
dot.edge('D4b', 'D4c')
dot.edge('D4c', 'D4d')

# ===== LAYER 5 – AUDIO CONTEXT =====
with dot.subgraph(name='cluster_audio') as c:
    c.attr(label='AUDIO ANALYSIS LAYER', style='rounded')
    c.node('A5b', 'Speech Detection')
    c.node('A5c', 'Whisper Transcription')
    c.node('A5d', 'Mood & Tempo')

dot.edge('A', 'A5b')
dot.edge('A5b', 'A5c')
dot.edge('A5b', 'A5d')

with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('A5c')
    s.node('A5d')

# ===== LAYER 6 – SELECTION =====
with dot.subgraph(name='cluster_selection') as c:
    c.attr(label='SELECTION LAYER', style='rounded')
    c.node('SEL', 'Density Selector')
    c.node('REP', 'Representative Frames')

dot.edge('SEL', 'REP')

# ===== LAYER 7 – LLM =====
with dot.subgraph(name='cluster_llm') as c:
    c.attr(label='LLM EXTRACTION LAYER', style='rounded')
    c.node('LLM', 'LLM Client\nextract()')
    c.node('OUT', 'Structured Output')

dot.edge('LLM', 'OUT')

# ===== CROSS-LAYER FLOWS =====
dot.edge('V', 'S2a')
dot.edge('S2c', 'C3a')
dot.edge('C3c', 'D4a')
dot.edge('D4d', 'SEL')

dot.edges([
    ('REP', 'LLM'),
    ('A5c', 'LLM'),
    ('A5d', 'LLM'),
    ('V', 'LLM')
])

# ===== IN-MEMORY RENDERING =====
img_bytes = dot.pipe(format='png')
Image(img_bytes)

# ===== SAVE PART (COMMENTED OUT) =====
# dot.format = 'pdf'
# dot.render('layered_ad_pipeline')
