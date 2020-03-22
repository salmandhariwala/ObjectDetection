import argparse

from imageai.Detection import ObjectDetection
from tabulate import tabulate

parser = argparse.ArgumentParser()

parser.add_argument("--input", help="file to detect", required=True)
parser.add_argument("--output", help="output directory", required=True)

args = parser.parse_args()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath('resnet50_coco_best_v2.0.1.h5')
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=args.input, output_image_path=args.output,
                                             extract_detected_objects=True)

print("-----------------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------------")

print()
print()

print(".....................................................................")
print("Following Objects were detected from input file : {}".format(args.input))

_objects = list()
_probability = list()

for eachObject in detections[0]:
    _objects.append(eachObject["name"])
    _probability.append(eachObject["percentage_probability"])

print(".....................................................................")
print(tabulate({"Objects": _objects, "Probability": _probability}, headers="keys", tablefmt="grid"))

print(".....................................................................")
print("Detected Image {0} \nExtracted object folder : {0}-object/".format(args.output))
print(".....................................................................")
