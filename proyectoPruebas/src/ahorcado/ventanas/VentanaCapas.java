package ahorcado.ventanas;

import java.awt.BorderLayout;
import java.awt.EventQueue;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

import ahorcado.ventanas.VentanaJuego;
import ahorcado.logica.CaracternovalidoException;
import ahorcado.logica.Fichero1;

import java.awt.FlowLayout;
import javax.swing.BoxLayout;
import java.awt.GridLayout;
import java.awt.Window;

import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JTextField;
import javax.swing.JButton;
import java.awt.event.ActionListener;
import java.util.HashMap;
import java.awt.event.ActionEvent;
import javax.swing.JPasswordField;

public class VentanaCapas extends JFrame {

	private JPanel contentPane;
	private JPanel panel_1;
	private JButton btnNewButton;
	private JButton btnSalir;

	public static JTextField textUsuario;
	public static JPasswordField textPassword;

	/**
	 * Launch the application.
	 */
	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					VentanaCapas frame = new VentanaCapas();
					frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	/**
	 * Create the frame.
	 */
	public VentanaCapas() {
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 450, 300);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(new BorderLayout(0, 0));
		
		JPanel panel = new JPanel();
		contentPane.add(panel, BorderLayout.NORTH);
		panel.setLayout(new GridLayout(1, 0, 0, 0));
		
		JLabel lblUsuario = new JLabel("Usuario:");
		panel.add(lblUsuario);
		
		textUsuario = new JTextField();
		panel.add(textUsuario);
		textUsuario.setColumns(10);
		
		JLabel lblNewLabel = new JLabel("password:");
		panel.add(lblNewLabel);
		
		textPassword = new JPasswordField();
		panel.add(textPassword);
		
		final Fichero1 data = new Fichero1();
		
		panel_1 = new JPanel();
		contentPane.add(panel_1, BorderLayout.CENTER);
		panel_1.setLayout(new GridLayout(1, 0, 0, 0));
		
		btnNewButton = new JButton("Entrar");
		btnNewButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				HashMap<String, String> usuarios = Fichero1.Gestion();
				String nombreUsuario = textUsuario.getText();

				if (usuarios.containsKey(nombreUsuario)) {
					String clave = textPassword.getText();
					try {
						ComprobarPass(clave);
						String passwordHashMap = usuarios.get(nombreUsuario);
						if (clave.equals(passwordHashMap)) {
							JOptionPane.showMessageDialog(VentanaCapas.this, "Bienvenido al juego");
							VentanaJuego Ej= new VentanaJuego();
							Ej.frame.setVisible(true);
						} else {
							JOptionPane.showMessageDialog(VentanaCapas.this, "Usuario o contraseña incorrectos");
						}
					} catch (CaracternovalidoException e1) {
						// TODO Auto-generated catch block
						JOptionPane.showMessageDialog(VentanaCapas.this, "Caracteres no validos");
					}

				} else {
					JOptionPane.showMessageDialog(VentanaCapas.this, "Usuario o contraseña incorrectos");
				}

		
			}
		});
		panel_1.add(btnNewButton);
		
		btnSalir = new JButton("Salir");
		btnSalir.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				System.exit(0);
			}
		});
		panel_1.add(btnSalir);
	}
	public void ComprobarPass(String pass) throws CaracternovalidoException {
		if (pass.contains("-") || pass.contains("/") || pass.contains(",") || pass.contains(".") || pass.contains("#")
				|| pass.contains(":") || pass.contains("^") || pass.contains("=") || pass.contains("&")
				|| pass.contains("(") || pass.contains(")") || pass.contains("$") || pass.contains("+")
				|| pass.contains("*"))
			throw new CaracternovalidoException();
	}
}
