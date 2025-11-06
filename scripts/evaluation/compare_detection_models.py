"""
Compare baseline and fine-tuned detection model performance.
"""
import json
from pathlib import Path

# Load results
baseline_path = Path(__file__).parent.parent.parent / 'outputs' / 'evaluation' / 'detection_baseline.json'
finetuned_path = Path(__file__).parent.parent.parent / 'outputs' / 'evaluation' / 'detection_finetuned.json'

with open(baseline_path, 'r') as f:
    baseline = json.load(f)

with open(finetuned_path, 'r') as f:
    finetuned = json.load(f)

# Extract overall stats
baseline_overall = baseline['statistics']['overall']
finetuned_overall = finetuned['statistics']['overall']
baseline_by_cond = baseline['statistics']['by_condition']
finetuned_by_cond = finetuned['statistics']['by_condition']

# Calculate deltas
delta_detection_rate = finetuned_overall['detection_rate'] - baseline_overall['detection_rate']
delta_avg_conf = finetuned_overall['avg_confidence'] - baseline_overall['avg_confidence']
delta_avg_detections = finetuned_overall['avg_detections_per_image'] - baseline_overall['avg_detections_per_image']

print("="*80)
print("DETECTION MODEL COMPARISON REPORT")
print("="*80)
print()
print("📊 Overall Performance")
print("-" * 80)
print(f"{'Metric':<30} {'Baseline':<15} {'Fine-tuned':<15} {'Delta':<15}")
print("-" * 80)
print(f"{'Detection Rate':<30} {baseline_overall['detection_rate']:<15.2%} {finetuned_overall['detection_rate']:<15.2%} {delta_detection_rate:+.2%}")
print(f"{'Avg Confidence':<30} {baseline_overall['avg_confidence']:<15.3f} {finetuned_overall['avg_confidence']:<15.3f} {delta_avg_conf:+.3f}")
print(f"{'Avg Detections/Image':<30} {baseline_overall['avg_detections_per_image']:<15.2f} {finetuned_overall['avg_detections_per_image']:<15.2f} {delta_avg_detections:+.2f}")
print()

# Per-condition comparison
print("📈 Performance by Condition")
print("-" * 80)

conditions = ['day_clear', 'day_rain', 'night_clear', 'night_rain']
for condition in conditions:
    if condition in baseline_by_cond and condition in finetuned_by_cond:
        base_cond = baseline_by_cond[condition]
        fine_cond = finetuned_by_cond[condition]
        
        delta_rate = fine_cond['detection_rate'] - base_cond['detection_rate']
        delta_conf = fine_cond['avg_confidence'] - base_cond['avg_confidence']
        
        print(f"\n{condition.upper().replace('_', ' ')}")
        print(f"  Detection Rate: {base_cond['detection_rate']:.2%} → {fine_cond['detection_rate']:.2%} ({delta_rate:+.2%})")
        print(f"  Avg Confidence: {base_cond['avg_confidence']:.3f} → {fine_cond['avg_confidence']:.3f} ({delta_conf:+.3f})")

print()
print("="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

# Decision logic
improvement_pct = delta_detection_rate * 100
if improvement_pct > 5:
    decision = "✅ STRONG IMPROVEMENT - Use fine-tuned model"
    recommendation = "The fine-tuned model shows significant improvement (+{:.1f}% detection rate). Deploy this model for production use.".format(improvement_pct)
elif improvement_pct >= 2:
    decision = "✅ MARGINAL IMPROVEMENT - Use fine-tuned model"
    recommendation = "The fine-tuned model shows modest improvement (+{:.1f}% detection rate). Recommended for production use.".format(improvement_pct)
else:
    decision = "⚠️ MINIMAL IMPROVEMENT - Consider baseline"
    recommendation = "The improvement is minimal (+{:.1f}% detection rate). Baseline model may be sufficient, but fine-tuned model still preferred for consistency.".format(improvement_pct)

print(f"\nDecision: {decision}")
print(f"\n{recommendation}")
print()
print(f"Key Improvements:")
print(f"  • Detection rate increased by {improvement_pct:+.1f}% ({baseline_overall['detection_rate']:.1%} → {finetuned_overall['detection_rate']:.1%})")
print(f"  • Average confidence improved by {delta_avg_conf:+.3f} ({baseline_overall['avg_confidence']:.3f} → {finetuned_overall['avg_confidence']:.3f})")
print(f"  • Most challenging condition (night_rain): {baseline_by_cond['night_rain']['detection_rate']:.1%} → {finetuned_by_cond['night_rain']['detection_rate']:.1%}")
print()
print("="*80)
