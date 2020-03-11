import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np

flags.DEFINE_boolean('debug', False, 'Debug flag')
flags.DEFINE_string('input', None, 'path to input .idl to filter')
flags.DEFINE_string('output', None, 'path to output results')
flags.DEFINE_integer('min_area', 0, 'Filter: Area restriction in bounding box')
flags.DEFINE_float('min_thresh', 0.0, 'Filter: Threshold restriction in bounding box')


def main(_argv):

    source_idl = FLAGS.input
    fp = open(source_idl)
    #create results file
    fout = open(FLAGS.output,"w")  
    
    filtered_boxes = 0
    print("Starting on idl: " + FLAGS.input)
    line = fp.readline()
    filtered = 0
    while line:
        #Get line ID and bounding box collection
        line_id,rhs = line.split(";",1)
        boxes = rhs.split(" (")
        #print(line)
        box_list = []
        score_list = []
        for i in range(len(boxes)):
            box = boxes[i].split("):",1)
            #Filter the first empty string
            if(len(box) == 1):
                continue
            box_list.append(box[0].split(", "))
            score_list.append((box[1].split(","))[0])

        box_int = np.zeros((len(box_list),4))
        score_float = np.zeros(len(score_list))
        for j in range(len(box_list)):
            score_float[j] = float(score_list[j])
            for i in range(4):
                box_int[j,i] = int(box_list[j][i])
        
        #We have now the int array.
        #Process it back to the idl
        output_line = line_id + "; "
        #Process each box
        for j in range(len(box_list)):
            if(FLAGS.min_area != 0): 
                box_area = box_int[j,2]*box_int[j,3]
                if(box_area < FLAGS.min_area):
                    filtered = filtered + 1
                    continue
            if(FLAGS.min_thresh != 0.0):
                if(score_float[j] < FLAGS.min_thresh):
                    filtered = filtered + 1
                    continue
            output_line = output_line + "({}, {}, {}, {}):{}, ".format(int(box_int[j,0]),
                                                                int(box_int[j,1]),
                                                                int(box_int[j,2]),
                                                                int(box_int[j,3]),
                                                                float(score_float[j]))
        #print(output_line)
        #Read a new line
        line = fp.readline()    
        fout.write(output_line + "\n")
       
    print("IDL: " + FLAGS.input + " DONE!") 
    print("Filtered: " + format(filtered)+ " Boxes.")
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
