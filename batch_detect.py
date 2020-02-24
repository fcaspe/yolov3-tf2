import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('dataset', None, 'path to dataset directory')
flags.DEFINE_string('output', None, 'path to output results')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def convert_box_to_img_size(img_size,box):
    int_box = np.zeros(4)
    int_box[0] = box[0]*img_size[1];
    int_box[1] = box[1]*img_size[0];
    int_box[2] = box[2]*img_size[1];
    int_box[3] = box[3]*img_size[0];
    
    #Convert to width and height
    int_box[2] = int_box[2] - int_box[0]
    int_box[3] = int_box[3] - int_box[1]
    
    return int_box.astype(int)


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    #OPEN THE IMAGE LIST
    #PARSE THE FIRST IMAGE FILENAME
    #OPEN THE IMAGE AS img_raw

    #open dataset 
    dataIn_file = FLAGS.dataset + "/dataIn.txt"
    fp = open(dataIn_file)
    #create results file
    fout = open(FLAGS.output,"w")  
    cnt = 1
    line = fp.readline()
    while line:
        #print("Line {}: {}".format(cnt, line.strip()))
        image_name = FLAGS.dataset + line
        image_name = image_name[0:len(image_name)-1]
        print(image_name)
        line = fp.readline()
        cnt += 1
        # GET THE RAW IMAGE
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)

    
        # IMAGE PROCESSING AFTER WE TAKE THE RAW ONE.
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)


        t2 = time.time()
        #logging.info('time: {}'.format(t2 - t1))
        
        #Log in idl
        output_line = "{0:c}{1:s}{2:c}; ".format(34,image_name,34)
        for i in range(nums[0]):
            int_box = convert_box_to_img_size(img_raw.shape,np.array(boxes[0][i]))
            output_line = output_line + "({}, {}, {}, {}):-{}, ".format(int_box[0],int_box[1],int_box[2],int_box[3],scores[0][i])
        fout.write(output_line + "\n")
        
        print("Persons detected:")
        for i in range(nums[0]):
            # Only process class 0 = person.
            if(int(classes[0][i]) == 0):
                print("\tPersonID " + format(i))
                print("\t\tSCORE: " + format(scores[0][i]))
            
                int_box = convert_box_to_img_size(img_raw.shape,np.array(boxes[0][i]))
                print("\t\tBOX: " + format(int_box))

    fp.close()
    fout.close()
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
