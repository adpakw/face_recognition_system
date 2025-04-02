from app.pipline_stages.people_detector import PeopleDetector

def main():
    people_detector = PeopleDetector()

    # people_detector.test_people_detector_img()

    people_detector.test_people_detector_video()

if __name__ == "__main__":
    main()



