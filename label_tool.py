import cv2
import numpy as np
import os

VIDEO = 'video'
IMAGE = 'image'
def show_video(name):
    # "name" include extension
    path = os.path.join(VIDEO, name)
    cap = cv2.VideoCapture(path)

    ret, frame_old = cap.read()
    
    # param for image cutting
    stack_frame = []
    num = 0

    while(1):
        ret, frame = cap.read()
        if frame is None:
            break
        
        diff_gray = cv2.cvtColor(cv2.absdiff(frame, frame_old), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(diff_gray, (5,5), 0)
        _, thres = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        erode = cv2.erode(thres, None, iterations=2)
        dilated = cv2.dilate(erode, None, iterations=2)

        frame_old = frame
        
        stack_frame.append(dilated)

        if len(stack_frame)==5:
            image = np.zeros_like(stack_frame[0])
            for frame in stack_frame:
                image = cv2.addWeighted(image, 0.8, frame, 1, 0)
            stack_frame.pop(0)
        
            # compute bounding box
            imgray = image.copy()
            imgray[imgray!=0] = 255
            contours = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x,y,w,h = cv2.boundingRect(contours[0])
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,255), 2)
            if contours[1] != []:
                crop_img = cv2.resize(image[y:y+h, x:x+w], (500,500), interpolation=cv2.INTER_AREA)
                cv2.imshow('a', crop_img)
                pause = cv2.waitKey(10) & 0xff
                if pause == 32:
                    save_image(crop_img, name, num)
                    num += 1
                elif pause == ord('b'):
                    break


    key = cv2.waitKey(0)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        end_program()
    elif key == ord('s'):
        cap.release()
        cv2.destroyAllWindows()
        save_video(name)
    elif key == ord('r'):
        cap.release()
        cv2.destroyAllWindows()
        remove_video(name)

    return 0

def save_video(name):
    # "name" include extension
    path1 = os.path.join(VIDEO, name)
    path2 = os.path.join('savedVideo', name)
    os.rename(path1, path2)
    return 0

def remove_video(name):
    # "name" include extension
    path = os.path.join(VIDEO, name)
    os.remove(path)
    return 0

def save_image(image,name,num):
    # "name" include extension
    path = os.path.join(IMAGE,"{}_{}.png".format(name, num))
    sign = cv2.imwrite(path, image)
    if sign:
        print('save {}'.format(path))
    return 0

def fetch_new_video():
    ld = os.listdir(VIDEO)
    print("remain: {}".format(len(ld)))
    if len(ld)==0:
        print('Done')
        end_program()
    return ld[0]

def end_program():
    exit(0)
    return 0

while(True):
    name = fetch_new_video()
    show_video(name)
