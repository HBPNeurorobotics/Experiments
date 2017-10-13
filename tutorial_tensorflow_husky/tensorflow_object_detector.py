##
## TensorFlow object detection, based on:
## https://github.com/GoogleCloudPlatform/tensorflow-object-detection-example
##
# internal TensorFlow state variables to persist
@nrp.MapVariable("detection_graph", initial_value=None)
@nrp.MapVariable("sess", initial_value=None)
@nrp.MapVariable("category_index", initial_value=None)
@nrp.MapVariable("bridge", initial_value=None)
@nrp.MapVariable("state", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable("target", initial_value=None, scope=nrp.GLOBAL)
# input generators to drive the Braitenberg brain
@nrp.MapSpikeSource("left_eye", nrp.brain.sensors[slice(0, 3, 2)], nrp.poisson)
@nrp.MapSpikeSource("right_eye", nrp.brain.sensors[slice(1, 4, 2)], nrp.poisson)
# subscribe to images from the robot
@nrp.MapRobotSubscriber("camera", Topic('/husky/camera', sensor_msgs.msg.Image))
# publish an annotated image as output of this transfer function
@nrp.Neuron2Robot(Topic('/detections', sensor_msgs.msg.Image))
def object_detection(t, detection_graph, sess, category_index, bridge, state, target, camera, left_eye, right_eye):

    # initialize the TensorFlow Object Detection session and store it as needed
    if detection_graph.value is None:

        # import TensorFlow in the NRP, update this path for your local installation
        try:
            import site
            site.addsitedir('<path to tensorflow venv>/lib/python2.7/site-packages')
            import tensorflow as tf
        except:
            clientLogger.info("Unable to import TensorFlow, did you change the path in the transfer function?")
            raise

        # configure Object Detection environment
        import sys

        # paths to saved model states, update these paths if different in your local installation
        MODEL_BASE = '/opt/models'
        sys.path.append(MODEL_BASE)
        sys.path.append(MODEL_BASE + '/object_detection')
        sys.path.append(MODEL_BASE + '/slim')

        PATH_TO_CKPT = '/opt/graph_def/frozen_inference_graph.pb'
        PATH_TO_LABELS = MODEL_BASE + '/object_detection/data/mscoco_label_map.pbtxt'

        # initialize the detection graph
        from utils import label_map_util
        detection_graph.value = tf.Graph()
        with detection_graph.value.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                sess.value = tf.Session(graph=detection_graph.value)

        # create internal label and category mappings
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=90,
                                                                    use_display_name=True)
        category_index.value = label_map_util.create_category_index(categories)

        # OpenCV bridge for ROS <-> CV image conversion
        from cv_bridge import CvBridge, CvBridgeError
        bridge.value = CvBridge()

        # initialized, start searching
        state.value = 'searching'

    # no image received yet, do nothing
    if camera.value is None:
        return

    # only run object detection if state is searching
    if state.value != 'searching':
        return

    # convert the ROS image to an OpenCV image and Numpy array
    cv_image = bridge.value.imgmsg_to_cv2(camera.value, "rgb8")
    numpy_image = np.expand_dims(cv_image, axis=0)

    # run the actual detection
    image_tensor = detection_graph.value.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.value.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.value.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.value.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.value.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.value.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: numpy_image})

    boxes, scores, classes, num_detections = map(
        np.squeeze, [boxes, scores, classes, num_detections])

    # annotate detections on the image
    from PIL import Image, ImageDraw
    pil_image = Image.fromarray(cv_image)
    detections = []
    for i in range(num_detections):

        # only accept high enough detection scores
        if scores[i] < 0.7: continue

        # log the detection at timestamp
        clientLogger.info(t, category_index.value[classes[i]]['name'], scores[i])
        detections.append(category_index.value[classes[i]]['name'])

        # annotate the image with boxes
        draw = ImageDraw.Draw(pil_image)
        im_width, im_height = pil_image.size
        ymin, xmin, ymax, xmax = boxes[i]
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=int(scores[i]*10)-4, fill='red')

        # if the object is our target goal, drive the husky towards its center
        if target.value == category_index.value[classes[i]]['name']:
            xmid = float(xmax - xmin) / 2.0
            clientLogger.info(xmid)
            left_eye.rate = 500 + (0 if xmid > 0.5 else 2000 * (0.5 - xmid))
            right_eye.rate = 500 + (0 if xmid < 0.5 else 2000 * (xmid - 0.5))
            clientLogger.info(left_eye.rate, right_eye.rate)

    # if we have detected the input area, wait for input
    if target.value is None and 'tv' in detections and 'chair' in detections:
        state.value = 'waiting'
        clientLogger.info('I see the tv! Waiting for user input.')

    # for a valid target that's not detected, reset search mode
    elif target.value is not None and target.value not in detections:
        left_eye.rate = 0
        right_eye.rate = 0

    # publish a ROS image with annotations
    import numpy
    return bridge.value.cv2_to_imgmsg(numpy.array(pil_image), "rgb8")
