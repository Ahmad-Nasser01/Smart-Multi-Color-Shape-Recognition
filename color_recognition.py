import cv2
import numpy as np

cap = cv2.VideoCapture(0)

prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0
alpha = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # نطاقات ألوان واسعة ومعدلة خصيصاً لتناسب إضاءة الغرف والواقع
    colors = {
        "Blue": ([85, 50, 40], [135, 255, 255], (255, 0, 0)),
        "Green": ([35, 40, 40], [85, 255, 255], (0, 255, 0)),
        "Red1": ([0, 70, 50], [12, 255, 255], (0, 0, 255)),
        "Red2": ([168, 70, 50], [180, 255, 255], (0, 0, 255)),
        "Yellow": ([15, 80, 80], [35, 255, 255], (0, 255, 255)),
        "White": ([0, 0, 190], [180, 40, 255], (200, 200, 200)),
        "Black": ([0, 0, 0], [180, 255, 40], (50, 50, 50))
    }

    counts = {"Blue": 0, "Green": 0, "Red": 0, "Yellow": 0, "White": 0, "Black": 0}

    for color_name, (lower, upper, bgr_color) in colors.items():
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        
        # تنظيف خفيف للقناع
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # تم تخفيض المساحة إلى 1200 لكي يلقط الأجسام بسهولة ولا يتجاهلها
            if area > 1200: 
                approx = cv2.approxPolyDP(largest_contour, 0.02 * cv2.arcLength(largest_contour, True), True)
                x, y, w, h = cv2.boundingRect(largest_contour)

                base_color = "Red" if "Red" in color_name else color_name
                counts[base_color] += 1

                # تحديد الشكل
                sides = len(approx)
                if sides == 3:
                    shape_name = "Triangle"
                elif sides == 4:
                    aspect_ratio = float(w) / h
                    shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
                else:
                    shape_name = "Object"

                # رسم الإطار والتسمية
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)
                label = f"{base_color} {shape_name}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)

    # لوحة العدادات في الزاوية
    cv2.rectangle(frame, (10, 10), (170, 125), (30, 30, 30), -1)
    cv2.putText(frame, "Dashboard", (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    y_offset = 45
    for color_name, count in counts.items():
        text = f"{color_name}: {count}"
        cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y_offset += 13

    cv2.imshow("Clean Multi-Color & Shape Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()