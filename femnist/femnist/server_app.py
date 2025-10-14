"""femnist: Flower / PyTorch server with metrics logging."""
import torch
import csv
from pathlib import Path
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from femnist.task import Net

# Path to save CSV
CSV_PATH = Path("fl_metrics.csv")

# Remove existing file (optional)
if CSV_PATH.exists():
    CSV_PATH.unlink()

# Write CSV headers
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "round", "client_id", "eval_loss", "eval_acc", 
        "avg_loss", "avg_acc"
    ])

# Custom aggregator for evaluation metrics
def evaluate_metrics_agg(records, weighting_key):
    """
    Aggregates client evaluation metrics and logs per-client + average to CSV.
    """
    total_weight = 0.0
    weighted_loss = 0.0
    weighted_acc = 0.0

    round_number = getattr(evaluate_metrics_agg, "round", 1)

    # Open CSV in append mode
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        for record in records:
            # Each record corresponds to one client's result
            metrics = list(record.metric_records.values())[0]
            client_id = metrics.get("client_id", f"client_{len(metrics)}")
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

    # Return aggregated metrics for Flower's FedAvg
    
    agg_metrics = MetricRecord({"eval_loss": avg_loss, "eval_acc": avg_acc})
    return agg_metrics

# ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Initialize global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg with custom evaluator to log metrics
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
    print(f"Metrics saved to {CSV_PATH}")
