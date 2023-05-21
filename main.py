import cv2
import numpy as np
import pygame
import random

# define a video capture object
vid = cv2.VideoCapture(0)
width = vid.get(3)
height = vid.get(4)
# Initialize Pygame
pygame.init()
window_width, window_height = int(width), int(height)
game_window = pygame.display.set_mode((window_width, window_height))

FONT = pygame.font.SysFont("Consolas", int(window_width / 20))
CLOCK = pygame.time.Clock()

ret, old_frame = vid.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

left_roi_x, left_roi_y, left_roi_width, left_roi_height = 0, 0, 100, window_width

right_roi_x, right_roi_y, right_roi_width, right_roi_height = window_width - 100, 0, 100, window_width
leftMiddleX, leftMiddleY, rightMiddleX, rightMiddleY = 1, 1, 1, 1

ball = pygame.Rect(window_width / 2 - 10, window_height / 2 - 10, 20, 20)
ball.center = (window_width / 2, window_height / 2)
ballXSpeed, ballYSpeed = 1, 1
player_score, opponent_score = 0, 0


def get_contours(flow):
    # Compute magnitude and angle of flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Threshold the magnitude to find the moving regions
    threshold = 20  # Adjust the threshold as needed
    moving_regions = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)[1]
    moving_regions = moving_regions.astype(np.uint8)  # Convert to uint8 format
    # Apply morphological operations to enhance the moving regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    moving_regions = cv2.morphologyEx(moving_regions, cv2.MORPH_CLOSE, kernel)
    # Find contours of the moving regions
    contours, _ = cv2.findContours(moving_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


# https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
def getPaddlePos(contours, oldX, oldY, drawOffset):
    newMiddleX = oldX
    newMiddleY = oldY
    allCenterX = []
    allCenterY = []
    largest_contour = None
    largest_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
    # Draw a square around the center of each moving object
    # for contour in contours:
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #     allCenterX.append(x + w // 2)
    #     allCenterY.append(y + h // 2)
    #     pygame.draw.rect(game_window, "green", [x+drawOffset, y, x+w+drawOffset, y+h], 1)
    if largest_contour is not None:
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        allCenterX.append(x)
        allCenterY.append(y)
        pygame.draw.rect(game_window, "green", [x + drawOffset, y, x + w + drawOffset, y + h], 1)

    if len(allCenterX) != 0:
        newMiddleX = np.sum(allCenterX) / len(allCenterX)
    if len(allCenterY) != 0:
        newMiddleY = np.sum(allCenterY) / len(allCenterY)
    return newMiddleX, newMiddleY


def getPaddle(middleX, middleY, xPos):
    # cv2.rectangle(frame, (xPos, int(paddle_y)), (xPos + int(paddle_width), int(paddle_y) + int(paddle_height)), (0, 0, 255),
    #               -1)
    return pygame.Rect(xPos, middleY, 10, 200)


while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if not ret:
        break

        # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cv2.destroyAllWindows()
            exit()

    # Render webcam frame onto the game window
    drawn_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for Pygame
    drawn_frame = cv2.flip(drawn_frame, 1)  # Flip the frame horizontally
    drawn_frame = cv2.resize(drawn_frame, (window_width, window_height))

    drawn_frame = cv2.rotate(drawn_frame,
                             cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate the frame by 90 degrees counterclockwise

    frame_surface = pygame.surfarray.make_surface(drawn_frame)
    game_window.blit(frame_surface, (0, 0))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(old_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Extract flow vectors within the ROI
    left_roi_flow = flow[left_roi_y:left_roi_y + left_roi_height, left_roi_x:left_roi_x + left_roi_width]
    left_contours = get_contours(left_roi_flow)
    leftMiddleX, leftMiddleY = getPaddlePos(left_contours, leftMiddleX, leftMiddleY, 0)
    leftPlayerX = 20
    leftPlayer = getPaddle(leftMiddleX, leftMiddleY, leftPlayerX)

    right_roi_flow = flow[right_roi_y:right_roi_y + right_roi_height, right_roi_x:right_roi_x + right_roi_width]
    right_contours = get_contours(right_roi_flow)
    rightMiddleX, rightMiddleY = getPaddlePos(right_contours, rightMiddleX, rightMiddleY, right_roi_x)
    rightPlayerX = 600
    rightPlayer = getPaddle(rightMiddleX, rightMiddleY, rightPlayerX)

    # # Display the result
    # cv2.imshow('Optical Flow', frame)
    if ball.y >= window_width-150:
        ballYSpeed = -1
    if ball.y <= 0:
        ballYSpeed = 1
    if ball.x <= 0:
        player_score += 1
        ball.center = (window_width / 2, window_height / 2)
        ballXSpeed, ballYSpeed = random.choice([1, -1]), random.choice([1, -1])
    if ball.x >= window_width:
        opponent_score += 1
        ball.center = (window_width / 2, window_height / 2)
        ballXSpeed, ballYSpeed = random.choice([1, -1]), random.choice([1, -1])

        # Check collision with left paddle
    if ball.colliderect(leftPlayer):
        ballXSpeed = 1  # Change ball's X-direction

        # Check collision with right paddle
    if ball.colliderect(rightPlayer):
        ballXSpeed = -1  # Change ball's X-direction

    ball.x += ballXSpeed * 15
    ball.y += ballYSpeed * 15

    player_score_text = FONT.render(str(player_score), True, "red")
    opponent_score_text = FONT.render(str(opponent_score), True, "red")

    game_window.blit(player_score_text, (window_width / 2 + 50, 50))
    game_window.blit(opponent_score_text, (window_width / 2 - 50, 50))

    pygame.draw.rect(game_window, "red", leftPlayer)
    pygame.draw.rect(game_window, "red", rightPlayer)
    pygame.draw.circle(game_window, "red", ball.center, 10)

    pygame.display.update()
    CLOCK.tick(300)

    old_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
