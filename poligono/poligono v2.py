import pygame
import random
import os
import sys

# Inizializzazione di Pygame
pygame.init()

# Definizione delle dimensioni della finestra di gioco
screen_width = 1200
screen_height = 720
screen = pygame.display.set_mode((screen_width, screen_height))

# Definizione dei colori
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Frame per secondo (FPS)
FPS = 60
clock = pygame.time.Clock()

# Caricamento delle immagini dalla cartella
image_folder = r"png_database"  # Assicurati che questa sia la cartella corretta con le immagini
image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]  # Filtra solo i file PNG
images = [pygame.image.load(os.path.join(image_folder, img)) for img in image_files]

# Scala tutte le immagini alla stessa dimensione (100x200)
scaled_images = [pygame.transform.scale(img, (100, 200)) for img in images]

# Classe per il bot (bersaglio con immagine)
class Bot:
    def __init__(self):
        self.width = 100
        self.height = 200  # Altezza e larghezza del bot
        self.image = random.choice(scaled_images)  # Assegna un'immagine casuale dal pool di immagini
        self.x = random.randint(0, screen_width - self.width)
        self.y = random.randint(100, screen_height - self.height - 100)
        self.speed = random.choice([2, 3, 4])  # Velocità randomica del bot
        self.direction = random.choice([-1, 1])  # Direzione iniziale (sinistra o destra)
        self.alive = True

    def move(self):
        if self.alive:
            self.x += self.speed * self.direction
            # Cambia direzione se raggiunge i bordi dello schermo
            if self.x <= 0 or self.x >= screen_width - self.width:
                self.direction *= -1

    def draw(self, screen):
        if self.alive:
            # Disegna l'immagine del bot
            screen.blit(self.image, (self.x, self.y))

    def is_hit(self, pos):
        # Controlla se il clic è dentro i confini dell'immagine del bot
        if self.alive and (self.x < pos[0] < self.x + self.width and self.y < pos[1] < self.y + self.height):
            self.alive = False
            return True
        return False

# Funzione principale del gioco
def game_loop():
    bots = [Bot() for _ in range(12)]  # Genera 15 bot con immagini diverse
    score = 0
    game_over = False

    while not game_over:
        screen.fill(WHITE)

        # Gestione degli eventi
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for bot in bots:
                    if bot.is_hit(pos):
                        score += 1

        # Muove e disegna i bot
        for bot in bots:
            bot.move()
            bot.draw(screen)

        # Controlla se tutti i bot sono stati eliminati
        if all(not bot.alive for bot in bots):
            game_over = True

        # Mostra il punteggio
        font = pygame.font.SysFont(None, 55)
        score_text = font.render(f'Score: {score}', True, GREEN)
        screen.blit(score_text, [10, 10])

        pygame.display.flip()
        clock.tick(FPS)

# Avvia il gioco
if __name__ == "__main__":
    game_loop()
