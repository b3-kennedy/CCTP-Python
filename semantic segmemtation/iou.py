
import numpy as np
import cv2


target = "E:/Users/Ben/Documents/CCTP/Splitaerialimaging/grassaccuracytests/maskgrass3.png"
prediction = "E:/Users/Ben/Documents/CCTP/AI/semantic segmemtation/Predictions/finalmasks/2pred.png"

targetimg = cv2.imread(target)

targetimg = cv2.resize(targetimg,(250,250))

# cv2.imshow("img",targetimg)

predictionimg = cv2.imread(prediction)

predictionimg = cv2.resize(predictionimg,(250,250))
# cv2.imshow("img",prediction)



intersection = np.logical_and(targetimg, predictionimg)
union = np.logical_or(targetimg, predictionimg)
iou_score = np.sum(intersection) / np.sum(union)

print(iou_score * 100)


#building scores: 68.34, 63.38, 57.1 average = 62.94
#road 62.03, 53.83, 69.7 average = 61.7
#tree 54.81, 80.35, 63.27 average = 66.14
#grass 52.35, 50, 69.61 average = 57.32