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
BLACK = (0, 0, 0)

# Frame per secondo (FPS)
FPS = 60
clock = pygame.time.Clock()

# Caricamento dell'immagine di sfondo
background_image = pygame.image.load(r"sfondo.jpg")
background_image = pygame.transform.scale(background_image, (screen_width, screen_height))

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
        self.x, self.y = self.generate_position()  # Posizione casuale dentro il rettangolo
        self.alive = True

    def generate_position(self):
        # Definire i limiti del rettangolo dove i bot possono apparire
        start_x = screen_width // 2
        line_length = screen_height // 2

        # Calcolare le coordinate dei punti finali delle linee a 45 gradi
        left_x = start_x - line_length
        right_x = start_x + line_length
        bottom_y = screen_height // 2  # Altezza della linea orizzontale

        # Calcolare i limiti di generazione
        x_range = (left_x + self.width, right_x - self.width)
        y_range = (100, bottom_y - self.height)

        # Generare una posizione casuale all'interno dei limiti
        x = random.randint(*x_range)
        y = random.randint(*y_range)

        return x, y

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

# Funzione per disegnare gli angoli a 45 gradi e una linea centrale dritta
def draw_lines():
    # Punto di partenza: centro in basso della finestra
    start_x = screen_width // 2
    start_y = screen_height
    
    # Lunghezza della linea (metà dell'altezza della finestra)
    line_length = screen_height // 2
    
    # Calcolare i punti finali delle linee a 45 gradi
    end_x_left = start_x - line_length  # Linea a sinistra: diminuisce x
    end_y_left = start_y - line_length  # Linea a sinistra: diminuisce y
    end_x_right = start_x + line_length  # Linea a destra: aumenta x
    end_y_right = start_y - line_length  # Linea a destra: diminuisce y

    # Disegna la linea a 45 gradi verso sinistra
    pygame.draw.line(screen, BLACK, (start_x, start_y), (end_x_left, end_y_left), 5)
    
    # Disegna la linea a 45 gradi verso destra
    pygame.draw.line(screen, BLACK, (start_x, start_y), (end_x_right, end_y_right), 5)
    
    # Disegna una linea verticale al centro fino alla metà della finestra
    pygame.draw.line(screen, BLACK, (start_x, start_y), (start_x, start_y - line_length), 5)
    
    # Disegna una linea orizzontale al centro
    pygame.draw.line(screen, BLACK, (0, screen_height // 2), (screen_width, screen_height // 2), 5)

# Funzione principale del gioco
def game_loop():
    current_bot = Bot()  # Inizializza un bot
    score = 0
    game_over = False

    while not game_over:
        # Disegna l'immagine di sfondo
        screen.blit(background_image, (0, 0))

        # Disegna le linee (due a 45 gradi, una verticale al centro e una orizzontale)
        draw_lines()

        # Gestione degli eventi
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if current_bot.is_hit(pos):
                    score += 1
                    current_bot = Bot()  # Crea un nuovo bot dopo aver cliccato quello corrente

        # Disegna il bot corrente
        current_bot.draw(screen)

        # Mostra il punteggio
        font = pygame.font.SysFont(None, 55)
        score_text = font.render(f'Score: {score}', True, GREEN)
        screen.blit(score_text, [10, 10])

        pygame.display.flip()
        clock.tick(FPS)

# Avvia il gioco
if __name__ == "__main__":
    game_loop()
