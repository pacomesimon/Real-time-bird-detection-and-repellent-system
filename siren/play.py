import pygame
import time
# see also: https://pypi.org/project/playsound/
pygame.mixer.init() # Initialize the mixer module
pygame.mixer.music.load("./siren/1653.wav") # Load your sound file



def play_buzzer():
    pygame.mixer.music.play() # Play the sound
    start_time = time.time()
    while pygame.mixer.music.get_busy() and (time.time() - start_time < 0.5):
        pygame.time.Clock().tick(10)
