from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def caption_similarity(caption1, caption2):
    emb1 = model.encode(caption1, convert_to_tensor=True)
    emb2 = model.encode(caption2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())

def relevance_to_goal(caption, goal):
    goal_emb = model.encode(goal, convert_to_tensor=True)
    cap_emb = model.encode(caption, convert_to_tensor=True)
    return float(util.cos_sim(goal_emb, cap_emb).item())
