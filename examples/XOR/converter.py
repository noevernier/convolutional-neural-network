import cv2
import os

image_folder = 'CNN/Examples/images'
video_name = 'video.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = ['frame'+str(i)+'.png' for i in range(100)]
print(images)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 60, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()