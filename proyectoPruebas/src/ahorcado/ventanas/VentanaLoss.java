package ahorcado.ventanas;
import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.Toolkit;

import javax.swing.JFrame;
import javax.swing.JLabel;

public class VentanaLoss extends JFrame { // Lo mismo que la clase VentanaWin pero se activa al perder
	
	private JLabel etiquetaPerdistes = new JLabel("     Lo siento, has perdido");
	private Toolkit t = Toolkit.getDefaultToolkit(); 
	private Image icono = t.getImage("Ahorcado0vidas.jpg"); 
	
	
	public VentanaLoss (){
		setSize(220,100);
		setLocationRelativeTo(null);
		setTitle("Fin del juego");
		setVisible(true);
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		setResizable(false);
		setIconImage(icono);
		
		add(etiquetaPerdistes);
		
	}

}
