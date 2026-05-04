import cv2

cap = cv2.VideoCapture('runs/detect/exp28/output.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow('Video Player', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): break  # Bấm Q để thoát

cap.release()
cv2.destroyAllWindows()