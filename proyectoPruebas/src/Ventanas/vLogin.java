package Ventanas;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;


import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;
import javax.swing.border.EmptyBorder;

public class vLogin extends JFrame {

	
	private static final long serialVersionUID = 1L;
	private JLabel lblUsuario;
	private JLabel lblContrasea;
	private JTextField textField_usu;
	private JPasswordField passwordField;
	
	
	public vLogin(){
		
		setTitle("Login");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 571, 458);
		JPanel contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(null);
		
		lblUsuario = new JLabel("Usuario:");
		lblUsuario.setBounds(76, 95, 85, 23);
		contentPane.add(lblUsuario);
		
		lblContrasea = new JLabel("Contrase\u00F1a:");
		lblContrasea.setBounds(76, 203, 85, 20);
		contentPane.add(lblContrasea);
		
		textField_usu = new JTextField();
		textField_usu.setBounds(193, 92, 287, 26);
		contentPane.add(textField_usu);
		textField_usu.setColumns(10);
		
		passwordField = new JPasswordField();
		passwordField.setBounds(193, 200, 287, 26);
		contentPane.add(passwordField);
		
		JButton btnOk = new JButton("OK");
		btnOk.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				
			}
		});
		btnOk.setBounds(365, 315, 115, 29);
		contentPane.add(btnOk);
		
		JButton btnSalir = new JButton("Salir");
		btnSalir.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				System.exit(0);
			}
		});
		btnSalir.setBounds(112, 315, 115, 29);
		contentPane.add(btnSalir);
		
		JPanel panel_usu = new JPanel();
		panel_usu.setBounds(15, 86, 46, 53);
		contentPane.add(panel_usu);
		
		JLabel label = new JLabel("");
		label.setIcon(new ImageIcon(vLogin.class.getResource("/proyectoPruebas/Places-user-identity-icon.png")));
		panel_usu.add(label);
		
		
		
		
		JPanel panel = new JPanel();
		panel.setBounds(15, 184, 46, 53);
		contentPane.add(panel);
		
		JLabel label_1 = new JLabel("");
		label_1.setIcon(new ImageIcon(vLogin.class.getResource("/proyectoPruebas/lock-icon.png")));
		panel.add(label_1);
		
		}
	
	
}
