# ChessMate

**ChessMate** is a prototype computer vision system that detects and logs chess moves from visual input. The system combines board detection, piece recognition, and game logic to convert a physical chessboard state into machine-readable move notation. This was developed as a small university prototype, using printed 2D icons to simplify piece detection.

---

## Table of Contents

- [Motivation](#motivation)  
- [Features](#features)  
- [Architecture](#architecture)  
- [Technologies Used](#technologies-used)  
- [Getting Started](#getting-started)  
- [Project Structure](#project-structure)  
- [Limitations & Future Work](#limitations--future-work)  
---

## Motivation

Chess is a classic structured problem that provides a controlled environment for testing computer vision and real-time decision logic. The goal of this project was to:

1. Detect a chessboard from an image or video feed.
2. Recognize the positions of chess pieces.
3. Infer moves made between consecutive board states.
4. Output a log of moves in standard notation (e.g., UCI or SAN).

This enables digitizing physical chess games without specialized hardware such as electronic chess boards or sensors.

---

## Features
- **Data Preprocessing**
- scripts/ contains scripts to help data generation for training the model as well as a grayscale randomizer which creates variations of images with different alpha and beta values following each a gaussian distribution, to simulate different lighting conditions for a more robust detection.

- **Chessboard Video Stream**  
  Uses image preprocessing to identify and warp the board into a normalized top-down view and show chessboard on display if not started in headless mode.

- **Piece Recognition**  
  Detection of pieces with a convolutional neural network (CNN) trained on own images. For the prototype, printed 2D icons were used to simplify detection.

- **Move Detection & Logging**  
  Compares board states to detect which piece moved where, and logs moves in a structured format.

---

## Architecture

The system performs the following high-level steps:

1. **Input Acquisition**  
   - Static images or frames from a video source are loaded.
   - Preprocessing adjusts brightness, contrast, and perspective.

2. **Board Localization**  
   - Locate chessboard with help of manual input.
   - Apply a perspective transform to get a rectified board.

3. **Grid Segmentation**  
   - Divide the rectified board into an 8×8 grid of tiles.

4. **Piece Detection**  
   - Classification and localization with CNN and deriving board position from coordinates and grid.

5. **State Comparison & Move Logging**  
   - Compare previous and current board states to infer the move made.
   - Log moves in human-readable and/or chess notation.

---

## Technologies Used

| Category | Tools & Libraries |
| -------- | ------------------ |
| Language | Python |
| Computer Vision | OpenCV |
| Model | SSD-Mobilenet |
| Data Handling | NumPy |
| Testing | Custom test cases / unit tests |

Dependencies are listed in `requirements.txt`.

---

## Getting Started

This project was build on a Jetson Nano for Linux environments and requires an NVIDIA GPU with CUDA installed and configured (refer to [https://docs.nvidia.com/cuda/cuda-installation-guide-linux/]) for instructions on how to do this.

### Prerequisites

(Instructions for Linux environments)

Install Python 3.8 or higher.

Then install dependencies:

```bash
pip install -r requirements.txt

# Clone Repository

git clone https://github.com/KianRK/chessmate.git
cd chessmate
python3 src/chessmate
```
## Project Structure

Typical output includes:

    Visualization of board segmentation and piece detection.

    A move log in standard chess notation (e.g., e2e4, g8f6).

    Optional debug images in test_output2/.

## Limitations & Future Work

This prototype was designed for ease of detection using printed 2D icons. For real-world chess sets:

    Piece detection should be improved using by training a model with more training data (especially needed for 3D pieces).


    Integration with a chess engine could enable move suggestions or validation.
