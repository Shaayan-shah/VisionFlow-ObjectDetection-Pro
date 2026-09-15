"""
VisionFlow-ObjectDetection-Pro
Sample Dataset & Benchmark Images Generator
Generates realistic high-resolution scenes with clear objects for instant testing.
"""

import os
import numpy as np
import cv2


def generate_sample_datasets(output_dir: str = "data/samples"):
    """Generate realistic test images for offline testing and immediate demo."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sample 1: Highway Traffic Scene (Cars, Trucks, Road)
    img_traffic = np.ones((720, 1280, 3), dtype=np.uint8)
    # Sky
    img_traffic[0:260, :] = [210, 190, 160] # Light blue-gray
    # Distant hills & buildings
    cv2.rectangle(img_traffic, (0, 220), (1280, 260), (120, 110, 95), -1)
    # Highway asphalt
    img_traffic[260:720, :] = [60, 60, 65]
    # Lane dividers
    for y in range(280, 720, 70):
        cv2.line(img_traffic, (420, y), (420, y + 40), (240, 240, 240), 4)
        cv2.line(img_traffic, (840, y), (840, y + 40), (240, 240, 240), 4)

    # Vehicle 1: Silver Sedan
    cv2.rectangle(img_traffic, (220, 420), (400, 560), (180, 185, 190), -1)
    cv2.rectangle(img_traffic, (250, 440), (370, 500), (90, 95, 100), -1) # Windshield
    cv2.circle(img_traffic, (260, 560), 22, (20, 20, 20), -1) # Wheel
    cv2.circle(img_traffic, (360, 560), 22, (20, 20, 20), -1)

    # Vehicle 2: Red Sports Car
    cv2.rectangle(img_traffic, (520, 360), (740, 490), (40, 40, 210), -1)
    cv2.rectangle(img_traffic, (560, 380), (690, 440), (70, 70, 80), -1)
    cv2.circle(img_traffic, (570, 490), 20, (20, 20, 20), -1)
    cv2.circle(img_traffic, (690, 490), 20, (20, 20, 20), -1)

    # Vehicle 3: Blue Cargo Truck
    cv2.rectangle(img_traffic, (890, 300), (1160, 520), (190, 100, 30), -1)
    cv2.rectangle(img_traffic, (890, 380), (970, 520), (160, 80, 20), -1) # Cabin
    cv2.circle(img_traffic, (930, 520), 24, (25, 25, 25), -1)
    cv2.circle(img_traffic, (1050, 520), 24, (25, 25, 25), -1)
    cv2.circle(img_traffic, (1120, 520), 24, (25, 25, 25), -1)

    path_traffic = os.path.join(output_dir, "highway_traffic.jpg")
    cv2.imwrite(path_traffic, img_traffic)

    # 2. Sample 2: Modern Tech Office Desk (Laptop, Monitor, Keyboard, Cup, Chair)
    img_office = np.ones((720, 1280, 3), dtype=np.uint8) * 240
    # Office wall & window
    img_office[0:380, :] = [245, 240, 235]
    # Desk Surface (Dark Wood)
    img_office[380:720, :] = [80, 110, 140]

    # Monitor
    cv2.rectangle(img_office, (480, 120), (880, 380), (30, 30, 35), -1)
    cv2.rectangle(img_office, (500, 140), (860, 360), (220, 140, 30), -1) # Screen glow
    cv2.rectangle(img_office, (660, 380), (700, 440), (80, 80, 85), -1) # Stand
    cv2.rectangle(img_office, (620, 440), (740, 460), (60, 60, 65), -1)

    # Laptop
    cv2.rectangle(img_office, (160, 360), (420, 540), (70, 70, 75), -1)
    cv2.rectangle(img_office, (180, 380), (400, 510), (180, 210, 230), -1)
    cv2.rectangle(img_office, (140, 540), (440, 580), (120, 120, 125), -1) # Base

    # Keyboard & Mouse
    cv2.rectangle(img_office, (520, 480), (820, 560), (45, 45, 50), -1) # Keyboard
    cv2.ellipse(img_office, (890, 520), (25, 40), 0, 0, 360, (50, 50, 55), -1) # Mouse

    # Coffee Cup
    cv2.circle(img_office, (1020, 480), 30, (230, 230, 230), -1)
    cv2.circle(img_office, (1020, 480), 24, (40, 60, 90), -1) # Dark coffee

    path_office = os.path.join(output_dir, "office_workspace.jpg")
    cv2.imwrite(path_office, img_office)

    # 3. Sample 3: Urban Pedestrian Street Crossing (People, Bicycles, Street)
    img_pedestrian = np.ones((720, 1280, 3), dtype=np.uint8)
    # Background shop facades
    img_pedestrian[0:340, :] = [180, 175, 170]
    # Sidewalk & crosswalk
    img_pedestrian[340:720, :] = [100, 105, 110]
    for x in range(100, 1200, 140):
        cv2.rectangle(img_pedestrian, (x, 480), (x + 80, 680), (230, 230, 235), -1)

    # Person 1 (Center Pedestrian)
    cv2.circle(img_pedestrian, (640, 380), 25, (170, 190, 210), -1) # Head
    cv2.rectangle(img_pedestrian, (615, 410), (665, 530), (180, 80, 50), -1) # Torso (Blue Jacket)
    cv2.rectangle(img_pedestrian, (620, 530), (640, 640), (40, 40, 45), -1) # Left leg
    cv2.rectangle(img_pedestrian, (645, 530), (665, 640), (40, 40, 45), -1) # Right leg

    # Person 2 (Left Pedestrian with Backpack)
    cv2.circle(img_pedestrian, (380, 410), 22, (160, 180, 200), -1)
    cv2.rectangle(img_pedestrian, (360, 435), (405, 540), (40, 140, 50), -1) # Green coat
    cv2.rectangle(img_pedestrian, (365, 540), (385, 630), (50, 50, 55), -1)
    cv2.rectangle(img_pedestrian, (390, 540), (410, 630), (50, 50, 55), -1)
    cv2.rectangle(img_pedestrian, (345, 450), (365, 510), (20, 30, 90), -1) # Backpack

    # Bicycle (Right side)
    cv2.circle(img_pedestrian, (920, 560), 38, (40, 40, 45), 4) # Wheel 1
    cv2.circle(img_pedestrian, (1060, 560), 38, (40, 40, 45), 4) # Wheel 2
    cv2.line(img_pedestrian, (920, 560), (990, 500), (20, 20, 180), 4) # Frame
    cv2.line(img_pedestrian, (990, 500), (1060, 560), (20, 20, 180), 4)
    cv2.line(img_pedestrian, (990, 500), (990, 450), (50, 50, 50), 4) # Handlebars

    path_pedestrian = os.path.join(output_dir, "pedestrian_street.jpg")
    cv2.imwrite(path_pedestrian, img_pedestrian)

    print(f"[VisionFlow] Generated 3 synthetic test datasets at: {output_dir}")
    return [path_traffic, path_office, path_pedestrian]


if __name__ == "__main__":
    generate_sample_datasets()
