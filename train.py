
# labelme.py
 
# import
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
 
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
 
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize
 
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
 
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
 
# Change it for your dataset's name
source="mydataset"

############################################################
#  Check to see if tensowflow-GPU working
############################################################
import tensorflow as tf
device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

############################################################
#  My Model Configurations (which you should change for your own task)
############################################################
 
class ModelConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Mmodel"
 
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2 # 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2# Background,
    # typically after labeled, class can be set from Dataset class
    # if you want to test your model, better set it corectly based on your trainning dataset
 
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
 
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
 
class InferenceConfig(ModelConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
############################################################
#  Dataset (My labelme dataset loader)
############################################################
 
class LabelmeDataset(utils.Dataset):
    # Load annotations
    # Labelme Image Annotator v = 3.16.7 
    # different version may have different structures
    # besides, labelme's annotation setup one file for 
    # one picture not for all pictures
    # and this annotations  are all dicts after Python json load 
    # {
    #   "version": "3.16.7",
    #   "flags": {},
    #   "shapes": [
    #     {
    #       "label": "balloon",
    #       "line_color": null,
    #       "fill_color": null,
    #       "points": [[428.41666666666674,  875.3333333333334 ], ...],
    #       "shape_type": "polygon",
    #       "flags": {}
    #     },
    #     {
    #       "label": "balloon",
    #       "line_color": null,
    #       "fill_color": null,
    #       "points": [... ],
    #       "shape_type": "polygon",
    #       "flags": {}
    #     },
    #   ],
    #   "lineColor": [(4 number)],
    #   "fillColor": [(4 number)],
    #   "imagePath": "10464445726_6f1e3bbe6a_k.jpg",
    #   "imageData": null,
    #   "imageHeight": 2019,
    #   "imageWidth": 2048
    # }
    # We mostly care about the x and y coordinates of each region
    def load_labelme(self, dataset_dir, subset):
        """
        Load a subset of the dataset.
        source: coustomed source id, exp: load data from coco, than set it "coco",
                it is useful when you ues different dataset for one trainning.(TODO)
                see the prepare func in utils model for details
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
 
        filenames = os.listdir(dataset_dir)
        jsonfiles,annotations=[],[]
        for filename in filenames:
            if filename.endswith(".json"):
                jsonfiles.append(filename)
                annotation = json.load(open(os.path.join(dataset_dir,filename)))
                # Insure this picture is in this dataset
                imagename = annotation['imagePath']
                if not os.path.isfile(os.path.join(dataset_dir,imagename)):
                    continue
                if len(annotation["shapes"]) == 0:
                    continue
                # you can filter what you don't want to load
                annotations.append(annotation)
                
        print("In {source} {subset} dataset we have {number:d} annotation files."
            .format(source=source, subset=subset,number=len(jsonfiles)))
        print("In {source} {subset} dataset we have {number:d} valid annotations."
            .format(source=source, subset=subset,number=len(annotations)))
 
        # Add images and get all classes in annotation files
        # typically, after labelme's annotation, all same class item have a same name
        # this need us to annotate like all "ball" in picture named "ball"
        # not "ball_1" "ball_2" ...
        # we also can figure out which "ball" it is refer to.
        labelslist = []
        for annotation in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            shapes = [] 
            classids = []
 
            for shape in annotation["shapes"]:
                # first we get the shape classid
                label = shape["label"]
                if labelslist.count(label) == 0:
                    labelslist.append(label)
                classids.append(labelslist.index(label)+1)
                shapes.append(shape["points"])
            
            # load_mask() needs the image size to convert polygons to masks.
            width = annotation["imageWidth"]
            height = annotation["imageHeight"]
            self.add_image(
                source,
                image_id=annotation["imagePath"],  # use file name as a unique image id
                path=os.path.join(dataset_dir,annotation["imagePath"]),
                width=width, height=height,
                shapes=shapes, classids=classids)
 
        print("In {source} {subset} dataset we have {number:d} class item"
            .format(source=source, subset=subset,number=len(labelslist)))
 
        for labelid, labelname in enumerate(labelslist):
            self.add_class(source,labelid,labelname)
 
    def load_mask(self,image_id):
        """
        Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not the source dataset you want, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != source:
            return super(self.__class__, self).load_mask(image_id)
 
        # Convert shapes to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["shapes"])], dtype=np.uint8)
        #printsx,printsy=zip(*points)
        for idx, points in enumerate(info["shapes"]):
            # Get indexes of pixels inside the polygon and set them to 1
            pointsy,pointsx = zip(*points)
            rr, cc = skimage.draw.polygon(pointsx, pointsy)
            mask[rr, cc, idx] = 1
        masks_np = mask.astype(np.bool)
        classids_np = np.array(image_info["classids"]).astype(np.int32)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return masks_np, classids_np
 
    def image_reference(self,image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == source:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
 
 
def train(dataset_train, dataset_val, model):
    """Train the model."""
    # Training dataset.
    dataset_train.prepare()
 
    # Validation dataset
    dataset_val.prepare()
 
    # *** This training schedule is an example. Update to your needs ***
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')
 
def test(model, image_path = None, video_path=None, savedfile=None):
    assert image_path or video_path
 
     # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Colorful
        import matplotlib.pyplot as plt
        
        _, ax = plt.subplots()
        visualize.get_display_instances_pic(image, boxes=r['rois'], masks=r['masks'], 
            class_ids = r['class_ids'], class_number=model.config.NUM_CLASSES,ax = ax,
            class_names=None,scores=None, show_mask=True, show_bbox=True)
        # Save output
        if savedfile == None:
            file_name = "test_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        else:
            file_name = savedfile
        plt.savefig(file_name)
        #skimage.io.imsave(file_name, testresult)
    elif video_path:
        pass
    print("Saved to ", file_name)
 
                
############################################################
#  Training and Validating
############################################################
 
if __name__ == '__main__':
    import argparse
 
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of your dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco', 'last' or 'imagenet'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=./logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to test and color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to test and color splash effect on')
    parser.add_argument('--classnum', required=False,
                        metavar="class number of your detect model",
                        help="Class number of your detector.")
    args = parser.parse_args()
 
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.image or args.video or args.classnum, \
            "Provide --image or --video and  --classnum of your model to apply testing"
 
 
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
 
    # Configurations
    if args.command == "train":
        config = ModelConfig()
        dataset_train, dataset_val = LabelmeDataset(), LabelmeDataset()
        dataset_train.load_labelme(args.dataset,"train")
        dataset_val.load_labelme(args.dataset,"val")
        config.NUM_CLASSES = len(dataset_train.class_info)
    elif args.command == "test":
        config = InferenceConfig()
        config.NUM_CLASSES = int(args.classnum)+1 # add backgrouond
        
    config.display()
 
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
 
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
 
    # Load weights
    print("Loading weights ", weights_path)
    if args.command == "train":
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes if we change the backbone?
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)
        # Train or evaluate
        train(dataset_train, dataset_val, model)
    elif args.command == "test":
        # we test all models trained on the dataset in different stage
        print(os.getcwd())
        filenames = os.listdir(args.weights)
        for filename in filenames:
            if filename.endswith(".h5"):
                print("Load weights from {filename} ".format(filename=filename))
                model.load_weights(os.path.join(args.weights,filename),by_name=True)
                savedfile_name = os.path.splitext(filename)[0] + ".jpg"
                test(model, image_path=args.image,video_path=args.video, savedfile=savedfile_name)
    else:
        print("'{}' is not recognized.Use 'train' or 'test'".format(args.command))


def get_display_instances_pic(image, boxes, masks, class_ids, class_number, ax,
    class_names=None,scores=None, show_mask=True, show_bbox=True,colors=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_number:a int about  How many class you have
    ax: used for save the pic
    class_names(optional): list of class names of the dataset
    scores: (optional) confidence scores for each box
    show_mask, show_bbox: To show masks and bounding boxes or not
    colors: (optional) An array or colors to use with each object
    """
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances detected *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
 
    # Generate random colors
    colors =random_colors(class_number)  if colors==None else colors
 
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
 
    masked_image = image.astype(np.uint32).copy()
 
    for i in range(N):
        class_id = class_ids[i]
        color = colors[class_id]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)
 
        # Label
        if class_names == None:
            label =  str(class_id) 
        else:
            label = class_names[class_id]
        score = scores[i] if scores is not None else None
        caption = "cls: {}  scr:{:.3f}".format(label, score) if score else "cls: {}".format(label)
 
        ax.text(x1, y1 + 8, caption, color='red', size=11, backgroundcolor="none")
 
        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)
 
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))