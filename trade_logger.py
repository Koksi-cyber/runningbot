import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

class TradeLogger:
    def __init__(self, log_path="trade_log.json"):
        self.log_path = Path(log_path)
        self.log_path.touch(exist_ok=True)
        self.trades: List[Dict] = self._load_log()

    def _load_log(self) -> List[Dict]:
        try:
            with self.log_path.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def _save_log(self):
        with self.log_path.open("w") as f:
            json.dump(self.trades, f, indent=2, default=str)

    def log_trade(self, timestamp: datetime, model: str, direction: str, result: str, pnl: float, probability: float):
        trade = {
            "timestamp": timestamp.isoformat(),
            "model": model,
            "direction": direction,
            "probability": probability,
            "result": result,
            "pnl": pnl,
        }
        self.trades.append(trade)
        self._save_log()

    def _get_week_start(self, dt: datetime) -> datetime:
        return dt - timedelta(days=dt.weekday(), hours=dt.hour, minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)

    def get_weekly_accuracy(self, model: str, now: Optional[datetime] = None) -> Dict:
        if now is None:
            now = datetime.utcnow()
        week_start = self._get_week_start(now)

        wins = 0
        losses = 0
        for trade in self.trades:
            if trade["model"] != model:
                continue
            ts = datetime.fromisoformat(trade["timestamp"])
            if ts >= week_start:
                if trade["result"] == "win":
                    wins += 1
                elif trade["result"] == "loss":
                    losses += 1

        total = wins + losses
        acc = wins / total if total > 0 else 0.0
        return {
            "week_start": week_start.isoformat(),
            "wins": wins,
            "losses": losses,
            "accuracy": round(acc, 4),
            "allowed": acc >= 0.75
        }

    def is_model_allowed(self, model: str, now: Optional[datetime] = None) -> bool:
        info = self.get_weekly_accuracy(model, now)
        return info["allowed"]

    def print_weekly_report(self):
        now = datetime.utcnow()
        for model in ["long", "short"]:
            acc = self.get_weekly_accuracy(model, now)
            print(f"[{model.upper()}] Week {acc['week_start']} | Accuracy: {acc['accuracy']*100:.2f}% | Wins: {acc['wins']} | Losses: {acc['losses']} | Allowed: {acc['allowed']}")
