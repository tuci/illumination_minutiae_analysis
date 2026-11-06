import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from read_images import read_images
from pre_processing import process_image_set
from minutiae_detection import detect_annotated_minutiae, detect_minutiae_automatic
from utils import compute_tp_fp_fn
from pyfing import LeaderParameters, Leader, Snfoe
from itertools import combinations
from matplotlib.ticker import MaxNLocator

# parse commandline arguments
parser = argparse.ArgumentParser(description='Evaluate contribution of new views')
parser.add_argument('--datafolder', help='Folder for images', nargs=1, type=str)
parser.add_argument('--savefolder', help='Folder for saved figures and files', nargs=1, type=str)
parser.add_argument('--prefix', help='Identifier for experiment/data type', nargs=1, type=str)
parser.add_argument('--max_block', help='Maximum block size for ridge period calculation. Default=128)', 
                    default=128, type=int)
parser.add_argument('--along_threshold', help='Distance threshold along the ridge direction. Default=20)',
                    default=20, type=int)
parser.add_argument('--across_threshold', help='Distance threshold across the ridge. Default = 10)',
                    default=10, type=int)
parser.add_argument('--minutia_quality_threshold', help='Minutiae detection threshold. Default=0.6', 
                    default=0.6, type=float)
parser.add_argument

args = parser.parse_args()

args.datafolder = '../Data/BK-AM-51-1-G/Aligned/'
args.savefolder = '../Figures/DirectionEval/BK-AM-51-1-G/'
args.prefix = 'bk-am-51-1-g'
args.max_block = 96
args.along_threshold = 20
args.across_threshold = 10
args.minutia_quality_threshold = 0.4

def plot_detection_distribution(dataframe, saveto):
    x = np.arange(len(dataframe))

    bar_width = 0.3
    gap = 0.08

    annot_x = x - (bar_width + gap)/2
    detect_x = x + (bar_width + gap)/2

    fig, ax = plt.subplots(figsize=(10, 6))

    # Annotated = Correct + Missed
    ax.bar(annot_x, dataframe["correct"], width=bar_width, color="tab:green",
           edgecolor="black", linewidth=1, label="Correct")

    ax.bar(annot_x, dataframe["miss"], width=bar_width, bottom=dataframe["correct"], 
           color="tab:red", edgecolor="black", linewidth=1, label="Miss")

    # Detected = Correct + Spurious
    ax.bar(detect_x, dataframe["correct"], width=bar_width, color="tab:green",
            edgecolor="black", linewidth=1)

    ax.bar(detect_x, dataframe["spurious"], width=bar_width, bottom=dataframe["correct"],
           color="tab:orange", edgecolor="black", linewidth=1, label="Spurious")

    # Direction labels

    ax.set_xticks(x)
    ax.set_xticklabels([])

    for i, direction in enumerate(dataframe["direction"]):
        ax.text(
            annot_x[i], -0.04, "Ann.",
            ha="center", va="top",
            transform=ax.get_xaxis_transform()
        )

        ax.text(
            detect_x[i], -0.04, "Det.",
            ha="center", va="top",
            transform=ax.get_xaxis_transform()
        )

        ax.text(
            x[i], -0.07, direction,
            ha="center", va="top",
            fontweight="bold",
            transform=ax.get_xaxis_transform()
        )

    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{saveto}_minutiae_detection_distribution.png')
    plt.close()

def find_closest_minutiae(reference, probe, along_threshold=20.0, across_threshold=5.0):

    candidates = []

    for i, ref in enumerate(reference):
        theta = ref.direction + np.pi /2 # Direction in radians
        for j, p in enumerate(probe):
            dx = p.x - ref.x
            dy = p.y - ref.y

            # Displacement along the minutia direction
            along = dx * np.cos(theta) + dy * np.sin(theta)

            # Displacement perpendicular to the direction
            across = -dx * np.sin(theta) + dy * np.cos(theta)

            # Elliptical distance
            distance = np.sqrt((along / along_threshold) ** 2 +
                               (across / across_threshold) ** 2)

            if distance <= 1.0:
                candidates.append((distance, i, j))

    candidates.sort(key=lambda x: x[0])

    matches = []
    used_reference = set()
    used_probe = set()

    for distance, i, j in candidates:
        if i in used_reference or j in used_probe:
            continue

        matches.append({"reference": reference[i],"probe": probe[j],
                        "distance": distance})

        used_reference.add(i)
        used_probe.add(j)

    return matches

# find n ew minutiae wrt to the reference minutiae set
def find_new_minutiae(reference_set, probe_set, along_threshold=20.0, across_threshold=5.0):

    matches = find_closest_minutiae(reference_set, probe_set, along_threshold, across_threshold)

    matched_ids = {id(match["probe"]) for match in matches}

    new_minutiae = [m for m in probe_set if id(m) not in matched_ids]

    return new_minutiae

def compute_gain_with_combination(views, combination, ref_detections, along_threshold=20.0, 
                                  across_threshold=10.0, use_annotations=False):
    merged_set_detections = ref_detections.copy()

    included_directions = []

    for c in combination:
        included_directions.append(views[c].direction)

        # Detections are already in common aligned coordinates
        view_detections = views[c].annotations if use_annotations else views[c].minutiae

        # Find minutiae not yet represented in the merged set
        new_detected_minutiae = find_new_minutiae(merged_set_detections, view_detections,
                                                  along_threshold, across_threshold)

        merged_set_detections.extend(new_detected_minutiae)

    total_gain = len(merged_set_detections) - len(ref_detections)
    lastview_gain = len(new_detected_minutiae) if combination else 0

    gain = {'NumCombi': len(combination) + 1, 'NumReference': len(ref_detections),
            'FinalSetSize': len(merged_set_detections), 'TotalGain': total_gain,
            'LastGain': lastview_gain, 'TotalGainRatio': total_gain / len(ref_detections) if ref_detections else 0,
            'LastGainRatio': lastview_gain / len(merged_set_detections) if merged_set_detections else 0,
            'Combination': included_directions}

    return gain

def main():
    imgfolder = args.datafolder
    acquisitions = read_images(imgfolder)

    for i, imgset in enumerate(acquisitions):
        imgset = process_image_set(imgset, histEq=True, target_ridge_distance=10,
                                   max_block=args.max_block)

    savefolder = args.savefolder
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
        os.mkdir(f'{savefolder}/Plots/')
        os.mkdir(f'{savefolder}/Figures/')
        os.mkdir(f'{savefolder}/CSVS/')

    params = LeaderParameters(minutia_quality_threshold=args.minutia_quality_threshold)
    ldr = Leader(parameters=params)
    snfoe = Snfoe()

    results = []

    for j, acquisition in enumerate(acquisitions):
        # Keep only the four cardinal illumination directions
        acquisition = [img for img in acquisition
                       if img.direction.lower() in ('north', 'south', 'east', 'west')]

        # Detect annotations and automatic minutiae once per image
        for img in acquisition:
            if getattr(img, 'annotations', None) is None:
                theta, _ = snfoe.run(img.normalisedROI, img.normalisedROIMask)
                img.annotations, _, _ = detect_annotated_minutiae(img, theta, min_area=1, 
                                                                  blur_ksize=False)

            if getattr(img, 'minutiae', None) is None:
                img.minutiae = detect_minutiae_automatic(img, ldr)

        # Single-view evaluation and combinations
        for i, img in enumerate(acquisition):
            direction = img.direction
            # Single-view detection performance
            crr, spr, mss = compute_tp_fp_fn(img.annotations, img.minutiae, dist_thresh=15, offset=(0, 0))

            baseline = {'Acquisition': j, 'NumCombi': 1, 'NumReference': len(img.annotations),
                        'FinalSetSize': len(img.annotations), 'TotalGain': 0, 'LastGain': 0,
                        'TotalGainRatio': 0, 'LastGainRatio': 0, 'NumReferenceDetected': len(img.minutiae),
                        'CorrectDetections': len(crr), 'MissedMinutiae': len(mss),
                        'SpuriousDetections': len(spr), 'Combination': None,
                        'AnnotationGain': 0, 'AnnotationFinalSetSize': len(img.annotations),
                        'DetectionGain': 0, 'DetectionFinalSetSize': len(img.minutiae)}
            baseline['Direction'] = direction
            baseline['CombiId'] = -1
            results.append(baseline)

            # Generate additional views
            views = list(range(len(acquisition)))
            views.remove(i)

            # Two-view combinations: evaluate both orders
            for c in views:
                combination = (c,)

                # Automatic detections
                detection_gain = compute_gain_with_combination(acquisition, combination, img.minutiae,
                                                               along_threshold=args.along_threshold,
                                                               across_threshold=args.across_threshold)

                # Annotations
                annotation_gain = compute_gain_with_combination(acquisition, combination, img.annotations,
                                                                along_threshold=args.along_threshold,
                                                                across_threshold=args.across_threshold,
                                                                use_annotations=True)

                gain_results = {'Acquisition': j, 'Direction': direction, 'Combination': [direction, acquisition[c].direction],
                                'NumCombi': 2, 'CombiId': c,
                                'AngularSeparation': (180 if {direction.lower(), acquisition[c].direction.lower()}
                                                       in ({'north', 'south'}, {'east', 'west'}) else 90),
                    'AnnotationGain': annotation_gain['TotalGain'],
                    'AnnotationFinalSetSize': annotation_gain['FinalSetSize'],
                    'DetectionGain': detection_gain['TotalGain'],
                    'DetectionFinalSetSize': detection_gain['FinalSetSize']}

                results.append(gain_results)

            # Three- and four-view combinations: fixed order
            for size in (3, 4):
                for selected in combinations(range(len(acquisition)), size):
                    if i != selected[0]:
                        continue
                    combination = selected[1:]
                    detection_gain = compute_gain_with_combination(acquisition, combination, img.minutiae,
                                                                   along_threshold=args.along_threshold,
                                                                   across_threshold=args.across_threshold)

                    annotation_gain = compute_gain_with_combination(acquisition, combination, img.annotations,
                                                                    along_threshold=args.along_threshold,
                                                                    across_threshold=args.across_threshold,
                                                                    use_annotations=True)

                    gain_results = {'Acquisition': j, 'Direction': direction,
                                    'Combination': [acquisition[k].direction for k in selected],
                                    'NumCombi': size, 'CombiId': '-'.join(map(str, selected)),
                                    'AngularSeparation': None, 'AnnotationGain': annotation_gain['TotalGain'],
                                    'AnnotationFinalSetSize': annotation_gain['FinalSetSize'],
                                    'DetectionGain': detection_gain['TotalGain'], 'DetectionFinalSetSize': detection_gain['FinalSetSize']}

                    results.append(gain_results)

    results = pd.DataFrame(results)

    # Save results
    # results.to_csv(f'{savefolder}/CSVS/{args.prefix}_minutiae_gain_evaluation.csv', index=False)

if __name__ == '__main__':
    main()


