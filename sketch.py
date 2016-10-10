import cv2
src = cv.imread('/Users/yidawang/Desktop/WechatIMG309.jpeg')
dst = cv2.stylization(src, sigma_s=60, sigma_r=0.07)
cv.imshow(dst)