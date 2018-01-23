package ahorcado.ventanas;

import java.awt.Image;
import java.awt.Toolkit;
import javax.swing.JFrame;

import ahorcado.general.LaminaJuego;

public class VentanaJuego extends JFrame {

	private Toolkit t = Toolkit.getDefaultToolkit(); // Sirve para poner la imagen
	private Image icono = t.getImage("Ahorcado6vidas.jpg"); // Ingreso la imagen que tengo en la carpeta del proyecto
	private LaminaJuego lamina = new LaminaJuego(); // Creamos el fondo de la ventana
	
	public VentanaJuego (){
		setSize(375,350); // Alto y ancho
		setLocationRelativeTo(null); // Centrar ventana
		setTitle("Ahorcado"); // Le pongo titulo
		setVisible(true); // La hago visible
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);// Al cerrar la ventana el programa no compila mas
		setIconImage(icono); // Le pongo la imagen
		setResizable(false); // Evito que se pueda modificar el tamaño de la pantalla
		
		add(lamina);// Le agrego lo que programe en Lamina
		

	}

}
