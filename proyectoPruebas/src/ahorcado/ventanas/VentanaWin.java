package ahorcado.ventanas;
import java.awt.BorderLayout;
import java.awt.Image;
import java.awt.Toolkit;

import javax.swing.JFrame;
import javax.swing.JLabel;

public class VentanaWin extends JFrame { // Ventana que se abre cuando el usuario gana
	
	private JLabel etiquetaGanastes = new JLabel("     Felicidades has ganado!"); // Creo la etiqueta que me muestra el mensaje
	private Toolkit t = Toolkit.getDefaultToolkit();  // Clase requerida para ponerle icono al jframe
	private Image icono = t.getImage("IconAhorcado.jpg"); //Creo la imagen y le pongo el mismo nombre que el del directorio donde esta guardada
	
	
	public VentanaWin (){ // Al inicializar esta clase pasa lo de abajo
		setSize(220,100); // Le asigno el tamaño
		setLocationRelativeTo(null); // La centro en el medio de la pantalla
		setTitle("Fin del juego"); // Le asigno el titulo
		setVisible(true); // La hago visible
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); // Le digo que al cerrarla no pase nada
		setResizable(false); // Evito que se pueda modificar el tamaño
		setIconImage(icono); // Le seteo el icono
		
		add(etiquetaGanastes); // Texto que indica que ganastes
		
		
		
	}

}
