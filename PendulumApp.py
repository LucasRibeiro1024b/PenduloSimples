import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

class PendulumApp:
    """
    Um aplicativo Python para simular e visualizar um pêndulo simples usando tkinter e matplotlib.
    A animação para quando o pêndulo atinge um estado de repouso.
    """
    def __init__(self, master):
        """
        Initializes the application, sets up the UI, and starts the simulation.
        """
        self.master = master
        master.title("Simulador de Pêndulo")
        master.geometry("1200x700")

        # --- Estado Interno ---
        # Aumentar o tempo de simulação para garantir que o pêndulo pare.
        self.t = np.linspace(0, 200, 2000)  # Pontos de tempo para a simulação (0 a 200s, 2000 frames)
        self.solution = None
        self.anim = None
        self.stop_frame = 0 # Frame em que a animação deve parar

        # --- Configuração da Interface ---
        self.setup_ui()
        self.start_animation()

    def setup_ui(self):
        """
        Creates and arranges all the UI widgets.
        """
        # Frames principais
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        config_frame = ttk.LabelFrame(main_frame, text="Configuração", padding="10")
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Novo frame para tabela de frequência
        freq_frame = ttk.LabelFrame(main_frame, text="Tabela de Execuções", padding="10")
        freq_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        anim_frame = ttk.LabelFrame(main_frame, text="Animação", padding="10")
        anim_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Widgets de Configuração ---
        ttk.Label(config_frame, text="Comprimento (L):").pack(pady=5)
        self.length_slider = ttk.Scale(config_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL)
        self.length_slider.set(1.0)
        self.length_slider.pack(pady=5, fill=tk.X)
        self.length_label = ttk.Label(config_frame, text=f"{self.length_slider.get():.2f} m")
        self.length_label.pack()
        self.length_slider.config(command=lambda v: self.length_label.config(text=f"{float(v):.2f} m"))

        ttk.Label(config_frame, text="Gravidade (g):").pack(pady=5)
        self.gravity_slider = ttk.Scale(config_frame, from_=1.0, to=25.0, orient=tk.HORIZONTAL)
        self.gravity_slider.set(9.81)
        self.gravity_slider.pack(pady=5, fill=tk.X)
        self.gravity_label = ttk.Label(config_frame, text=f"{self.gravity_slider.get():.2f} m/s²")
        self.gravity_label.pack()
        self.gravity_slider.config(command=lambda v: self.gravity_label.config(text=f"{float(v):.2f} m/s²"))

        ttk.Label(config_frame, text="Ângulo Inicial (θ₀):").pack(pady=5)
        self.angle_slider = ttk.Scale(config_frame, from_=1, to=179, orient=tk.HORIZONTAL)
        self.angle_slider.set(45.0)
        self.angle_slider.pack(pady=5, fill=tk.X)
        self.angle_label = ttk.Label(config_frame, text=f"{self.angle_slider.get():.1f}°")
        self.angle_label.pack()
        self.angle_slider.config(command=lambda v: self.angle_label.config(text=f"{float(v):.1f}°"))
        
        ttk.Label(config_frame, text="Amortecimento (b):").pack(pady=5)
        self.damping_slider = ttk.Scale(config_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL)
        self.damping_slider.set(0.1)
        self.damping_slider.pack(pady=5, fill=tk.X)
        self.damping_label = ttk.Label(config_frame, text=f"{self.damping_slider.get():.2f}")
        self.damping_label.pack()
        self.damping_slider.config(command=lambda v: self.damping_label.config(text=f"{float(v):.2f}"))

        restart_button = ttk.Button(config_frame, text="Reiniciar Animação", command=self.start_animation)
        restart_button.pack(pady=20, fill=tk.X)

        # --- Canvas de Animação ---
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=anim_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # --- Tabela de Frequência ---
        columns = ("Comprimento (L)", "Gravidade (g)", "Ângulo Inicial (θ₀)", "Amortecimento (b)", "Tempo Final (s)")
        self.freq_table = ttk.Treeview(freq_frame, columns=columns, show="headings", height=20)
        for col in columns:
            self.freq_table.heading(col, text=col)
            self.freq_table.column(col, width=110, anchor=tk.CENTER)
        self.freq_table.pack(fill=tk.BOTH, expand=True)

        # Scrollbar para a tabela
        scrollbar = ttk.Scrollbar(freq_frame, orient="vertical", command=self.freq_table.yview)
        self.freq_table.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def pendulum_ode(self, y, t, b, g, L):
        """
        Define a equação diferencial para um pêndulo amortecido.
        """
        theta, omega = y
        dydt = [omega, -b * omega - (g / L) * np.sin(theta)]
        return dydt

    def solve_pendulum(self):
        """
        Resolve a EDO do pêndulo e encontra o frame de parada.
        """
        L = self.length_slider.get()
        g = self.gravity_slider.get()
        b = self.damping_slider.get()
        theta0_rad = np.radians(self.angle_slider.get())
        y0 = [theta0_rad, 0.0]
        
        self.solution = odeint(self.pendulum_ode, y0, self.t, args=(b, g, L))

        # --- Lógica da Condição de Parada ---
        # Define limiares para ângulo (theta) e velocidade angular (omega)
        angle_threshold = 0.01  # radianos (aprox. 0.6 graus)
        omega_threshold = 0.01  # radianos/s

        # Encontra os índices onde o pêndulo está "parado"
        stopped_indices = np.where(
            (np.abs(self.solution[:, 0]) < angle_threshold) &
            (np.abs(self.solution[:, 1]) < omega_threshold)
        )[0]

        if len(stopped_indices) > 0:
            # Pega o primeiro frame onde a condição é satisfeita
            self.stop_frame = stopped_indices[0]
        else:
            # Se nunca parar (ex: sem amortecimento), anima tudo
            self.stop_frame = len(self.t) - 1
            
        # Garante que o stop_frame não seja zero, a menos que comece parado.
        if self.stop_frame == 0 and theta0_rad > angle_threshold:
             self.stop_frame = len(self.t) - 1

    def animate(self, i):
        """
        Função de animação chamada para cada frame.
        """
        L = self.length_slider.get()
        theta = self.solution[i, 0]
        x = L * np.sin(theta)
        y = -L * np.cos(theta)

        self.ax.clear()
        self.ax.set_xlim(-L*1.1, L*1.1)
        self.ax.set_ylim(-L*1.1, 0.1)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid()
        self.ax.plot([0, x], [0, y], 'o-', lw=2, color='blue')
        self.ax.plot(0, 0, 's', markersize=10, color='red')
        self.ax.set_title(f"Tempo: {self.t[i]:.2f}s", fontsize=12)

        # Detecta o último frame da animação para registrar na tabela
        if i == self.stop_frame:
            self.register_frequency_row(self.t[i])

        return self.ax,

    def register_frequency_row(self, tempo_final):
        """
        Adiciona uma linha na tabela de execuções.
        """
        L = self.length_slider.get()
        g = self.gravity_slider.get()
        theta0 = self.angle_slider.get()
        b = self.damping_slider.get()
        self.freq_table.insert(
            "", "end",
            values=(
                f"{L:.2f}", f"{g:.2f}", f"{theta0:.1f}", f"{b:.2f}", f"{tempo_final:.2f}"
            )
        )

    def start_animation(self):
        """
        Inicia ou reinicia a animação.
        """
        if self.anim is not None and hasattr(self.anim, "event_source") and self.anim.event_source is not None:
            self.anim.event_source.stop()
        
        self.solve_pendulum()
        
        self.anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=range(self.stop_frame + 1), # Anima somente até o frame de parada
            interval=20, # Intervalo menor para uma animação mais fluida
            blit=False,
            repeat=False
        )
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PendulumApp(root)
    root.mainloop()
