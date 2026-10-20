"""
Chest X-Ray Disease Detection - Visualization Tool

This script generates GradCAM heatmap visualizations for chest X-ray
analysis. It shows disease localization, detection confidence, and
provides relevant clinical information.

Usage:
    python visualize.py <image_path>
    python visualize.py <image_path> --output <output_path>
    python visualize.py <image_path> --disease Atelectasis
"""

import argparse
import os

import cv2
from disease_info import DISEASE_INFO, format_report_section
from gradcam import (
    GradCAM,
    create_all_diseases_report,
    create_healthy_report,
    get_activation_region,
)
from image_processor import load_image, process_xray, resize_image
from model_handler import PATHOLOGIES, TARGET_DISEASES, THRESHOLDS, XRayModel


def generate_report(image_path: str, output_dir: str | None = None,
                   target_disease: str | None = None, show_all: bool = False):
    """
    Generate a complete diagnostic visualization report.

    Args:
        image_path: Path to the chest X-ray image
        output_dir: Directory to save output files (default: same as input)
        target_disease: Specific disease to analyze (default: highest detection)
        show_all: Generate visualizations for all detected diseases
    """

    # Validate input
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return

    # Set output directory - always save to results folder
    if output_dir is None:
        output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\nAnalyzing: {image_path}")
    print("-" * 50)

    # Load and preprocess image
    print("Loading model...")
    model = XRayModel()
    model.load_model()

    # Process image for model
    print("Processing image...")
    img_tensor = process_xray(image_path)

    # Get predictions
    print("Running inference...")
    predictions = model.predict_diseases(img_tensor)

    # Print prediction summary
    print("\nPrediction Results:")
    print("-" * 30)

    detected = []
    for disease, data in predictions.items():
        status = "[DETECTED]" if data['detected'] else "[negative]"
        print(f"  {disease}: {data['probability']:.1%} (threshold: {data['threshold']:.1%}) {status}")
        if data['detected']:
            detected.append((disease, data['probability']))

    # Sort detected by probability
    detected.sort(key=lambda x: x[1], reverse=True)

    # Load original image for visualization (WITHOUT normalization for display)
    original = load_image(image_path)
    original_display = resize_image(original, 224)  # For display - no normalization

    if not detected:
        print("\nNo pathologies detected above threshold.")
        print("Note: This does not rule out disease. Clinical correlation is required.")

        # Create HEALTHY report - use original_display (not normalized)
        print("\nCreating healthy report...")
        healthy_report = create_healthy_report(original_display, predictions, img_size=480)
        report_path = os.path.join(output_dir, f"{base_name}_healthy_report.png")
        cv2.imwrite(report_path, cv2.cvtColor(healthy_report, cv2.COLOR_RGB2BGR))
        print(f"Healthy report saved: {report_path}")
        return

    # Initialize GradCAM
    print("\nGenerating GradCAM visualizations...")
    gradcam = GradCAM(model.model)

    # Generate heatmaps for ALL detected diseases (for combined view)
    all_heatmaps = {}
    all_probabilities = {}
    all_locations = {}

    for disease_name, probability in detected:
        print(f"\nProcessing: {disease_name}")

        try:
            heatmap = gradcam.generate_for_disease(img_tensor, disease_name, PATHOLOGIES)
            all_heatmaps[disease_name] = heatmap
            all_probabilities[disease_name] = probability

            # Analyze activation region
            region = get_activation_region(heatmap, threshold=0.5)
            location = region['location'] if region['found'] else "Not localized"
            all_locations[disease_name] = location

            if region['found']:
                print(f"  Primary focus: {location}")
                print(f"  Affected area: {region['area_percentage']:.1f}% of image")

        except Exception as e:
            print(f"  Error generating heatmap: {e}")
            continue

    # Create and save COMBINED visualization with all diseases
    if all_heatmaps:
        # Create SINGLE comprehensive report image with all DETECTED diseases
        print("\nCreating comprehensive disease report...")
        print(f"  Showing {len(all_heatmaps)} DETECTED diseases (above threshold)")

        report_img = create_all_diseases_report(
            original_display,  # Use original without normalization
            all_heatmaps,
            all_probabilities,
            DISEASE_INFO,
            thresholds=THRESHOLDS,  # Pass thresholds for display
            img_size=480  # HD quality
        )

        report_path = os.path.join(output_dir, f"{base_name}_disease_report.png")
        cv2.imwrite(report_path, cv2.cvtColor(report_img, cv2.COLOR_RGB2BGR))
        print(f"Disease report saved: {report_path}")

        # Show NOT detected diseases
        not_detected = [d for d in TARGET_DISEASES if d not in all_heatmaps]
        if not_detected:
            print(f"\n  NOT Detected (below threshold): {', '.join(not_detected)}")

    # Print disease information for each detected disease
    for disease_name, probability in detected:
        if disease_name in all_locations:
            print("\n" + "=" * 60)
            report = format_report_section(disease_name, probability, all_locations[disease_name])
            print(report)
            print("=" * 60)

    # Cleanup
    gradcam.remove_hooks()

    # Text report disabled - all info is in the image report
    # report_path = os.path.join(output_dir, f"{base_name}_report.txt")
    # ... text report generation removed ...

    print("\nAnalysis complete.")


def _get_timestamp():
    """Get current timestamp string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate GradCAM visualization for chest X-ray analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize.py chest_xray.png
  python visualize.py chest_xray.png --output ./results

Output files:
  - <image>_all_findings.png  : Combined visualization with all diseases
  - <image>_report.txt        : Detailed text report
        """
    )

    parser.add_argument(
        "image",
        help="Path to chest X-ray image"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output directory for results (default: same as input image)"
    )

    args = parser.parse_args()

    generate_report(
        image_path=args.image,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
