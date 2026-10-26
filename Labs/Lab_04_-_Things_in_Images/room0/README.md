# Lab 04: the labelled set

`boxes.json` holds the 351 true boxes (chair 84, couch 60, dining table 107, vase 55, book 45) of 59
keyframes of the Replica `room0` scene, in the COCO format. The boxes were made from the scene's per-pixel
class maps, so touching objects of one class share a box, and an object cut in two by something in front
of it has two.

The images are not here. They are frames of the Replica dataset (Straub et al., "The Replica Dataset: A
Digital Replica of Indoor Spaces", arXiv 1906.05797), rendered as in NICE-SLAM: the files
`results/frameNNNNNN.jpg` (1200 x 680) of the `room0` folder of the Replica package which pySLAM's README
links. Lab 04 (Setting Up) says how to get them.

From the `pyslam` folder, with `R` set to your `room0` folder:

```bash
pixi run -e semantics models-detection --set ../Labs/Lab_04_-_Things_in_Images/room0 --images $R
pixi run -e semantics object-detection --method YOLO --evaluate ../Labs/Lab_04_-_Things_in_Images/room0 --images $R
```

The first command checks that the 59 images are there, and that they are the files the boxes were made on.
