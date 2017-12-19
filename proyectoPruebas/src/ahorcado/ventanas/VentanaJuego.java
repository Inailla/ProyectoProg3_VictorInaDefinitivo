package ahorcado.ventanas;
import java.awt.EventQueue;

import javax.swing.JFrame;
import javax.swing.JSplitPane;
import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.util.ArrayList;

import javax.swing.JPanel;
import javax.swing.JTextPane;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JTextField;
import javax.swing.JButton;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import javax.swing.JLabel;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeEvent;

public class VentanaJuego {

	public JFrame frame;
	private JTextField textNuevo;
	private String palabraNormal = "normal";
	private String palabraAsterisco = "";
	private JTextField textAcertar;
	private int contador =0;
	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					VentanaJuego window = new VentanaJuego();
					window.frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**
	 * Create the application.
	 */
	public VentanaJuego() {
		initialize();
	}

	/**
	 * Initialize the contents of the frame.
	 */
	private void initialize() {
		frame = new JFrame();
		frame.setBounds(100, 100, 450, 300);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().setLayout(new GridLayout(0, 1, 0, 0));
		
		JSplitPane splitPane = new JSplitPane();
		splitPane.setResizeWeight(0.4);
		frame.getContentPane().add(splitPane);
		
		JPanel panel = new JPanel();
		splitPane.setLeftComponent(panel);
		panel.setLayout(new GridLayout(3, 2, 0, 0));
		
		textNuevo = new JTextField();
		panel.add(textNuevo);
		textNuevo.setColumns(10);
		
		JButton btnNewButton = new JButton("Aceptar");

		panel.add(btnNewButton);
		
		textAcertar = new JTextField();
		panel.add(textAcertar);
		textAcertar.setColumns(10);
		
		JButton btnAcertar = new JButton("Acertar");
		panel.add(btnAcertar);
		
		JLabel lblTienesVidas = new JLabel("Empiezas con 8 vidas");
		panel.add(lblTienesVidas);
		
		JLabel lblPalabra = new JLabel("palabra");
		lblPalabra.addPropertyChangeListener(new PropertyChangeListener() {
			public void propertyChange(PropertyChangeEvent arg0) {
				if(palabraAsterisco.equals(palabraNormal)){
					JOptionPane.showMessageDialog(frame, "Palabra acertada");
				VentanaCaraFeliz VcF= new VentanaCaraFeliz();
				VcF.setVisible(true);
				}
				}
		});
		splitPane.setRightComponent(lblPalabra);
		
		btnAcertar.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				contador++;
				if(contador==8){
					VentanaCaraTriste VcT= new VentanaCaraTriste();
					VcT.setVisible(true);
				}
				char letraAcertar = textAcertar.getText().charAt(0);
				char[] letrasAsterisco = palabraAsterisco.toCharArray();
				char[] letrasNormal = palabraNormal.toCharArray();
				int cont = 0;
				
				for(char letra : letrasNormal){
					if(letra==letraAcertar)
						letrasAsterisco[cont] = letra;
					cont++;
				}
				palabraAsterisco = "";
				for(char letra : letrasAsterisco)
					palabraAsterisco += letra; 
				
				lblPalabra.setText(palabraAsterisco);
							}
			

		});
		btnNewButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				palabraNormal = textNuevo.getText();
				palabraAsterisco = "";
				int cont = 0;
				for(char letra : palabraNormal.toCharArray()){
					if (cont==0 || cont==palabraNormal.length()-1)
						palabraAsterisco += letra;
					else
						palabraAsterisco += "*";
					cont++;
				}
				lblPalabra.setText(palabraAsterisco);
			}
		});
	}

}
