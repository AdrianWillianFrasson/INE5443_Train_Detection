import cv2 as cv
import numpy as np
# import pytesseract

# -----------------------------------------------------------------------------
IMAGE_FOLDER = "./dataset/images/00/"
IMAGE_BACKGROUND = "./dataset/empty.png"
EXPORT_FOLDER = "./export/"


# -----------------------------------------------------------------------------
def draw_bounding_box(trains, image):
    for train in trains:
        x, y, w, h = train["bounding"]
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv.putText(image, train["name"], (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def show_log(trains):
    log = np.zeros([900, 210, 3], dtype=np.uint8)
    log.fill(255)

    cv.putText(log, f"N. de Trens: {len(trains)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    for i, train in enumerate(trains):
        text = f"{train["name"]}: {train["status"]} - {train["station"]}"
        cv.putText(log, text, (10, 75 + 20*i), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv.imwrite(f"{EXPORT_FOLDER}9_log.png", log)
    cv.imshow("log", log)


def train_classification(trains_boundings, image):
    trains = list()

    for index, bounding in enumerate(trains_boundings):
        x, y, w, h = bounding

        name = f"{index + 1}"
        # name = pytesseract.image_to_string(image[y:y+h, x:x+w], lang="eng", config="--psm 6")

        x_center = x + w / 2
        y_center = y + h / 2

        # Line 1
        if 450 <= x_center <= 6400 and 300 <= y_center <= 615:
            status = "Line 1"

        # Line 2
        elif 450 <= x_center <= 6400 and 1170 <= y_center <= 1600:
            status = "Line 2"

        # Parked
        elif 0 <= x_center <= 400 and 300 <= y_center <= 1600:
            status = "Parked"

        # OffLine
        else:
            status = "Off Line"

        station = ""

        if status == "Line 1" or status == "Line 2":

            if x_center <= 1750:
                if 515 <= x_center <= 555:
                    station = "TUC"

                elif 840 <= x_center <= 880:
                    station = "PIG"

                elif 1025 <= x_center <= 1065:
                    station = "JPA"

                elif 1385 <= x_center <= 1425:
                    station = "SAN"

                elif 1500 <= x_center <= 1540:
                    station = "CDU"

                elif 1635 <= x_center <= 1675:
                    station = "TTE"

            elif x_center <= 3150:
                if 1830 <= x_center <= 1870:
                    station = "PPQ"

                elif 1940 <= x_center <= 1975:
                    station = "TRD"

                elif 2455 <= x_center <= 2495:
                    station = "LUZ"

                elif 2615 <= x_center <= 2655:
                    station = "BTO"

                elif 2715 <= x_center <= 2755:
                    station = "PSE"

                elif 3020 <= x_center <= 3060:
                    station = "LIB"

            elif x_center <= 4900:
                if 3255 <= x_center <= 3295:
                    station = "JQM"

                elif 3420 <= x_center <= 3455:
                    station = "VGO"

                elif 3750 <= x_center <= 3790:
                    station = "PSO"

                elif 4045 <= x_center <= 4085:
                    station = "ANR"

                elif 4580 <= x_center <= 4615:
                    station = "VMN"

                elif 4795 <= x_center <= 4830:
                    station = "SCZ"

            else:
                if 4990 <= x_center <= 5025:
                    station = "ARV"

                elif 5240 <= x_center <= 5280:
                    station = "SAU"

                elif 5440 <= x_center <= 5475:
                    station = "JUD"

                elif 5630 <= x_center <= 5665:
                    station = "CON"

                elif 6000 <= x_center <= 6040:
                    station = "JAB"

        trains.append({
            "name": name,
            "status": status,
            "station": station,
            "bounding": bounding,
        })

    return trains


def train_detection(image, background):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    background_gray = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

    cv.imwrite(f"{EXPORT_FOLDER}3_base_gray.png", background_gray)
    cv.imwrite(f"{EXPORT_FOLDER}4_test_gray.png", image_gray)

    output = cv.subtract(image_gray, background_gray)
    cv.imwrite(f"{EXPORT_FOLDER}5_subtract.png", output)

    output_names = cv.threshold(output, 80, 255, cv.THRESH_BINARY)[1]
    cv.imwrite(f"{EXPORT_FOLDER}6_threshold.png", output_names)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))

    output = cv.dilate(output_names, kernel)
    cv.imwrite(f"{EXPORT_FOLDER}7_dilate.png", output)

    contours, _ = cv.findContours(output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    boundings = list(map(lambda contour: cv.boundingRect(contour), contours))

    trains_boundings = list()

    for bounding in boundings:
        w, h = bounding[2], bounding[3]

        # Vertical trains
        if 37 <= w <= 44 and 136 <= h <= 170:
            trains_boundings.append(bounding)

        # Horizontal trains
        elif 85 <= w <= 87 and 65 <= h <= 67:
            trains_boundings.append(bounding)

    cv.imshow("output2", output)
    return train_classification(trains_boundings, output_names)


def image_processing(image_path):
    image = cv.imread(image_path)
    background = cv.imread(IMAGE_BACKGROUND)

    cv.imwrite(f"{EXPORT_FOLDER}1_base.png", background)
    cv.imwrite(f"{EXPORT_FOLDER}2_test.png", image)

    trains = train_detection(image, background)

    draw_bounding_box(trains, image)
    show_log(trains)

    cv.imwrite(f"{EXPORT_FOLDER}8_bounding_box.png", image)
    cv.imshow("output1", image)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    image_index = 1

    cv.namedWindow("output1", cv.WINDOW_NORMAL)
    cv.namedWindow("output2", cv.WINDOW_NORMAL)
    cv.namedWindow("log", cv.WINDOW_NORMAL)

    while True:
        image_processing(f"{IMAGE_FOLDER}{image_index:03d}.png")

        key = cv.waitKey(0)

        if key == ord("q"):
            image_index -= 1

        elif key == ord("e"):
            image_index += 1

        elif key == ord("w"):
            cv.destroyAllWindows()
            break
