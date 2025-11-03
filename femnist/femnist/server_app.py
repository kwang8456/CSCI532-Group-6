"""femnist: Flower / PyTorch server with PID-based client exclusion."""
import torch
import csv
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from femnist.task import Net

# ServerApp
app = ServerApp()

class PIDFedAvg(FedAvg):
    """FedAvg with PID-based malicious client detection and exclusion."""
    
    def __init__(
        self,
        *args,
        pid_threshold: float = 2.0,
        kp: float = 1.0,
        ki: float = 0.05,
        kd: float = 0.5,
        csv_path: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pid_threshold = pid_threshold
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.csv_path = csv_path
        
        # Track PID components for each client
        self.client_distances: Dict[str, List[float]] = {}  # D(t) history
        self.client_integrals: Dict[str, float] = {}  # Integral term
        self.excluded_clients: set = set()  # Permanently excluded clients
        self.current_round = 0
    
    def _array_to_numpy(self, arr):
        """Convert Flower Array object to numpy array."""
        # Try .numpy() method first (common in array-like objects)
        if hasattr(arr, 'numpy'):
            return arr.numpy()
        # Fallback to np.array()
        return np.array(arr)
        
    def aggregate_train(self, grid, arrays_records, **kwargs):
        """Aggregate training results with PID-based filtering."""
        self.current_round += 1
        
        if len(arrays_records) == 0:
            return None, {}
        
        # Step 1: Calculate client model weights as numpy arrays
        client_weights = {}
        client_ids = []
        
        for i, record in enumerate(arrays_records):
            client_id = f"client_{i}"
            client_ids.append(client_id)
            
            # Extract arrays from Message or ArrayRecord
            if hasattr(record, 'content'):
                # It's a Message object, get arrays from content
                arrays = record.content.get('arrays', record.content.get('array_records', {}))
                if hasattr(arrays, 'arrays'):
                    weights = [arr for arr in arrays.arrays.values()]
                else:
                    weights = [arr for arr in arrays.values()]
            elif hasattr(record, 'arrays'):
                # It's an ArrayRecord
                weights = [arr for arr in record.arrays.values()]
            else:
                # Fallback - try to get arrays attribute
                weights = [arr for arr in record.values()]
            
            client_weights[client_id] = weights
        
        # Step 2: Calculate centroid (mean of all client weights)
        if len(client_weights) > 0:
            # Get all weight arrays, flatten and concatenate
            all_weights_flat = []
            for weights in client_weights.values():
                # Convert Array objects to numpy arrays before flattening
                flat = np.concatenate([self._array_to_numpy(w).flatten() for w in weights])
                all_weights_flat.append(flat)
            
            centroid = np.mean(all_weights_flat, axis=0)
            
            # Step 3: Calculate PID scores for each client
            pid_scores = {}
            clients_to_exclude = []
            
            for client_id, weights in client_weights.items():
                if client_id in self.excluded_clients:
                    continue  # Skip already excluded clients
                
                # Calculate L2 distance to centroid
                # Convert Array objects to numpy arrays before flattening
                flat_weights = np.concatenate([self._array_to_numpy(w).flatten() for w in weights])
                distance = np.linalg.norm(flat_weights - centroid)
                
                # Normalize by square root of dimension for scale-invariance
                # This makes distances comparable across different model sizes
                if len(flat_weights) > 0:
                    distance = distance / np.sqrt(len(flat_weights))
                
                # Initialize client history if first time
                if client_id not in self.client_distances:
                    self.client_distances[client_id] = []
                    self.client_integrals[client_id] = 0.0
                
                # Store current distance
                self.client_distances[client_id].append(distance)
                
                # Calculate PID score
                if len(self.client_distances[client_id]) == 1:
                    # First round: PID = Kp * D(t)
                    pid_score = self.kp * distance
                else:
                    # Subsequent rounds: full PID formula
                    # Proportional term
                    p_term = self.kp * distance
                    
                    # Integral term (sum of all past distances)
                    self.client_integrals[client_id] += distance
                    i_term = self.ki * self.client_integrals[client_id]
                    
                    # Derivative term (change from last round)
                    d_term = self.kd * (distance - self.client_distances[client_id][-2])
                    
                    pid_score = p_term + i_term + d_term
                
                pid_scores[client_id] = pid_score
                
                # Check threshold
                if pid_score > self.pid_threshold:
                    clients_to_exclude.append(client_id)
                    self.excluded_clients.add(client_id)
            
            # Log exclusions
            if self.csv_path:
                self._log_exclusions(clients_to_exclude, pid_scores)
            
            print(f"\n[Round {self.current_round}] PID Exclusion Summary:")
            print(f"  Clients evaluated: {len(pid_scores)}")
            print(f"  Clients excluded this round: {len(clients_to_exclude)}")
            print(f"  Total excluded: {len(self.excluded_clients)}")
            if clients_to_exclude:
                print(f"  Excluded: {clients_to_exclude}")
            
            # Debug: Show top 5 highest PID scores
            sorted_scores = sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top 5 PID scores: {[(cid, f'{score:.6f}') for cid, score in sorted_scores[:5]]}")
            print(f"  Threshold: {self.pid_threshold:.6f}")
            
            # Step 4: Filter out excluded clients from aggregation
            filtered_records = []
            filtered_indices = []
            for i, record in enumerate(arrays_records):
                client_id = client_ids[i]
                if client_id not in self.excluded_clients:
                    filtered_records.append(record)
                    filtered_indices.append(i)
            
            print(f"  Clients included in aggregation: {len(filtered_records)}")
            
            # If too many clients excluded (less than 2 remaining), include some least suspicious ones
            # This prevents the system from becoming completely ineffective
            min_clients_needed = 2
            if len(filtered_records) < min_clients_needed:
                if len(filtered_records) == 0:
                    print("  WARNING: All clients excluded! Including least suspicious clients based on PID scores.")
                else:
                    print(f"  WARNING: Only {len(filtered_records)} client(s) remaining! Including more least suspicious clients.")
                
                # Get PID scores for all clients this round, sort by score (ascending)
                if pid_scores:
                    sorted_by_score = sorted(pid_scores.items(), key=lambda x: x[1])
                    # Ensure we have at least min_clients_needed, but prefer lower PID scores
                    num_to_include = max(min_clients_needed, min(len(arrays_records) // 2, len(arrays_records) - 1))
                    
                    # Include clients with lowest PID scores, but exclude already-permanently-excluded ones
                    included_client_ids = set()
                    for cid, score in sorted_by_score:
                        if len(included_client_ids) >= num_to_include:
                            break
                        # Don't re-include permanently excluded clients unless we have no choice
                        if len(included_client_ids) < min_clients_needed or cid not in self.excluded_clients:
                            included_client_ids.add(cid)
                    
                    # Build filtered records with least suspicious clients
                    filtered_records = []
                    for i, record in enumerate(arrays_records):
                        client_id = client_ids[i]
                        if client_id in included_client_ids:
                            filtered_records.append(record)
                    
                    print(f"  Included {len(filtered_records)} least suspicious clients: {sorted([cid for cid in included_client_ids])}")
                else:
                    # Fallback: use all if no PID scores available
                    filtered_records = arrays_records
                    print("  No PID scores available, using all clients.")
        else:
            filtered_records = arrays_records
        
        # Step 5: Perform standard FedAvg on filtered clients
        return super().aggregate_train(grid, filtered_records, **kwargs)
    
    # this removes excluded clients from evaluation
    # def aggregate_evaluate(self, grid, arrays_records, **kwargs):
    #     """Aggregate evaluation results, excluding permanently excluded clients."""
    #     if len(arrays_records) == 0:
    #         return None, {}
        
    #     # Filter out excluded clients from evaluation
    #     filtered_records = []
    #     for record in arrays_records:
    #         # Extract client ID from the record
    #         metrics = list(record.metric_records.values())[0]
    #         client_id = f"client_{metrics.get('client_id', 'unknown')}"
            
    #         # Only include non-excluded clients
    #         if client_id not in self.excluded_clients:
    #             filtered_records.append(record)
        
    #     # Use parent's aggregate_evaluate on filtered records
    #     return super().aggregate_evaluate(grid, filtered_records, **kwargs)
    
    def _log_exclusions(self, excluded_this_round: List[str], pid_scores: Dict[str, float]):
        """Log client exclusions to CSV."""
        exclusion_csv = self.csv_path.parent / f"exclusions_{self.csv_path.stem}.csv"
        
        # Create file with headers if it doesn't exist
        if not exclusion_csv.exists():
            with open(exclusion_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "client_id", "pid_score", "excluded", "permanently_excluded"])
        
        # Append data
        with open(exclusion_csv, "a", newline="") as f:
            writer = csv.writer(f)
            for client_id, pid_score in pid_scores.items():
                excluded = client_id in excluded_this_round
                permanently_excluded = client_id in self.excluded_clients
                writer.writerow([
                    self.current_round,
                    client_id,
                    f"{pid_score:.6f}",
                    excluded,
                    permanently_excluded
                ])


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    
    # Get attack parameters
    attack: bool = context.run_config.get("attack-mode", False)
    flip_pct: int = context.run_config.get("flip-pct", 0)
    
    # Get PID parameters (with defaults)
    use_pid: bool = context.run_config.get("use-pid", False)
    pid_threshold: float = context.run_config.get("pid-threshold", 2.0)
    kp: float = context.run_config.get("kp", 1.0)
    ki: float = context.run_config.get("ki", 0.05)
    kd: float = context.run_config.get("kd", 0.5)
    
    # Setup CSV path
    base = Path("results") / ("attack" if attack else "no_attack")
    base.mkdir(parents=True, exist_ok=True)
    
    strategy_name = "pid" if use_pid else "conventional"
    csv_path = base / f"fl_metrics_{strategy_name}_{flip_pct}.csv"
    
    # Remove existing file
    if csv_path.exists():
        csv_path.unlink()
    
    # Write CSV headers
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round", "client_id", "eval_loss", "eval_acc", 
            "avg_loss", "avg_acc"
        ])
    
    # Custom aggregator for evaluation metrics
    # def evaluate_metrics_agg(records, weighting_key):
    #     """Aggregates client evaluation metrics and logs per-client + average to CSV."""
    #     total_weight = 0.0
    #     weighted_loss = 0.0
    #     weighted_acc = 0.0
    #     round_number = getattr(evaluate_metrics_agg, "round", 1)
        
    #     with open(csv_path, "a", newline="") as f:
    #         writer = csv.writer(f)
    #         for record in records:
    #             metrics = list(record.metric_records.values())[0]
    #             client_id = metrics.get("client_id", f"client_{len(metrics)}")
    #             loss = float(metrics.get("eval_loss", 0.0))
    #             acc = float(metrics.get("eval_acc", 0.0))
    #             ne = float(metrics.get(weighting_key, 1))
    #             writer.writerow([round_number, client_id, loss, acc, "", ""])
    #             weighted_loss += loss * ne
    #             weighted_acc += acc * ne
    #             total_weight += ne
            
    #         avg_loss = weighted_loss / total_weight if total_weight else 0.0
    #         avg_acc = weighted_acc / total_weight if total_weight else 0.0
    #         writer.writerow([round_number, "avg", "", "", avg_loss, avg_acc])
        
    #     evaluate_metrics_agg.round = round_number + 1
        
    #     agg_metrics = MetricRecord({"eval_loss": avg_loss, "eval_acc": avg_acc})
    #     return agg_metrics
    
    def evaluate_metrics_agg(records, weighting_key):
        total_weight = 0.0
        weighted_loss = 0.0
        weighted_acc = 0.0
        round_number = getattr(evaluate_metrics_agg, "round", 1)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for record in records:
                metrics = list(record.metric_records.values())[0]
                client_id = f"client_{metrics.get('client_id', 'unknown')}"
                
                # skip excl client
                if hasattr(strategy, "excluded_clients") and client_id in strategy.excluded_clients:
                    continue
                
                loss = float(metrics.get("eval_loss", 0.0))
                acc = float(metrics.get("eval_acc", 0.0))
                ne = float(metrics.get(weighting_key, 1))
                writer.writerow([round_number, client_id, loss, acc, "", ""])
                weighted_loss += loss * ne
                weighted_acc += acc * ne
                total_weight += ne

            avg_loss = weighted_loss / total_weight if total_weight else 0.0
            avg_acc = weighted_acc / total_weight if total_weight else 0.0
            writer.writerow([round_number, "avg", "", "", avg_loss, avg_acc])

        evaluate_metrics_agg.round = round_number + 1
        return MetricRecord({"eval_loss": avg_loss, "eval_acc": avg_acc})


    # Initialize global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    
    # Choose strategy based on use_pid flag
    if use_pid:
        print(f"\n{'='*60}")
        print(f"Starting PID-Enhanced Federated Learning")
        print(f"PID Parameters: Kp={kp}, Ki={ki}, Kd={kd}, Threshold={pid_threshold}")
        print(f"{'='*60}\n")
        
        strategy = PIDFedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            evaluate_metrics_aggr_fn=evaluate_metrics_agg,
            pid_threshold=pid_threshold,
            kp=kp,
            ki=ki,
            kd=kd,
            csv_path=csv_path,
        )
    else:
        print(f"\n{'='*60}")
        print(f"Starting Conventional Federated Learning (No PID)")
        print(f"{'='*60}\n")
        
        strategy = FedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            evaluate_metrics_aggr_fn=evaluate_metrics_agg,
        )
    
    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )
    
    # Save final model
    print("\nSaving final model to disk...")
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    print(f"Metrics saved to {csv_path}")
    if use_pid:
        exclusion_csv = csv_path.parent / f"exclusions_{csv_path.stem}.csv"
        print(f"Exclusion log saved to {exclusion_csv}")