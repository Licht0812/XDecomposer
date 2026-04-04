
import os
import torch
import numpy as np
import pickle
import logging
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from dara.benchmark_loader import XRDBaseDataset, process_pattern

class CosineMatcher:
    """
    XRD Pattern Matcher using Cosine Similarity against a Reference Bank.
    """
    def __init__(
        self, 
        singlephase_db_path: str, 
        crystal_ids: List[int],
        xrd_length: int = 3500,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_path: Optional[str] = None
    ):
        self.db_path = singlephase_db_path
        self.crystal_ids = crystal_ids
        self.xrd_length = xrd_length
        self.device = device
        self.cache_path = cache_path or os.path.join(singlephase_db_path, "reference_bank.pt")
        
        self.reference_bank = None # [N, L]
        self.id_map = None # Index -> Crystal ID
        
        self._load_or_build_bank()

    def _load_or_build_bank(self):
        if os.path.exists(self.cache_path):
            logging.info(f"Loading reference bank from {self.cache_path}...")
            data = torch.load(self.cache_path, map_location="cpu")
            self.reference_bank = data['bank'].to(self.device)
            self.id_map = data['id_map']
            return

        logging.info(f"Building reference bank from {len(self.crystal_ids)} IDs...")
        dataset = XRDBaseDataset(self.db_path, xrd_length=self.xrd_length)
        
        bank = []
        id_map = []
        
        for cid in tqdm(self.crystal_ids, desc="Indexing reference patterns"):
            patterns = dataset.get_crystal_patterns(cid, max_samples=1)
            if not patterns:
                continue
            
            tensor = process_pattern(patterns[0], self.xrd_length, norm_method="max")
            bank.append(tensor)
            id_map.append(cid)
            
        self.reference_bank = torch.stack(bank).to(self.device)
        self.id_map = id_map
        
        # Normalize for cosine similarity
        self.reference_bank = torch.nn.functional.normalize(self.reference_bank, p=2, dim=1)
        
        logging.info(f"Saving reference bank to {self.cache_path}...")
        torch.save({'bank': self.reference_bank.cpu(), 'id_map': self.id_map}, self.cache_path)
        self.reference_bank = self.reference_bank.to(self.device)

    def match(self, predicted_pattern: torch.Tensor, topk: int = 10) -> List[Tuple[int, float]]:
        """
        Match a predicted pattern against the reference bank.
        Returns list of (crystal_id, similarity)
        """
        if predicted_pattern.dim() == 1:
            predicted_pattern = predicted_pattern.unsqueeze(0)
            
        predicted_pattern = predicted_pattern.to(self.device)
        predicted_pattern = torch.nn.functional.normalize(predicted_pattern, p=2, dim=1)
        
        # [1, L] @ [L, N] -> [1, N]
        similarities = torch.mm(predicted_pattern, self.reference_bank.t()).squeeze(0)
        
        scores, indices = torch.topk(similarities, k=min(topk, len(self.id_map)))
        
        results = []
        for s, idx in zip(scores.cpu().tolist(), indices.cpu().tolist()):
            results.append((self.id_map[idx], s))
            
        return results
