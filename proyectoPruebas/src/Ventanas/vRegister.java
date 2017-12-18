package Ventanas;

import java.awt.BorderLayout;
import java.awt.EventQueue;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

import BaseDatos.Db;
import datos.Usuario;

import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JTextField;
import javax.swing.JPasswordField;
import javax.swing.JRadioButton;
import javax.swing.JButton;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

public class vRegister extends JFrame {

	private JPanel contentPane;
	private JTextField textField_usu;
	private JTextField textField_3;
	private JPasswordField passwordField;
	private JPasswordField passwordField_1;
	private String url = System.getProperty("user.dir")+"/sample1.db";

	
	public vRegister() {
		setTitle("Register");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 613, 506);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(null);
		
		JLabel lblUsuario = new JLabel("Usuario:");
		lblUsuario.setBounds(56, 74, 69, 20);
		contentPane.add(lblUsuario);
		
		JLabel lblContrasea = new JLabel("Contrase\u00F1a:");
		lblContrasea.setBounds(56, 156, 106, 20);
		contentPane.add(lblContrasea);
		
		JLabel lblComprobarContrasea = new JLabel("Comprobar");
		lblComprobarContrasea.setBounds(56, 224, 87, 20);
		contentPane.add(lblComprobarContrasea);
		
		JLabel lblEmail = new JLabel("Email:");
		lblEmail.setBounds(56, 315, 69, 20);
		contentPane.add(lblEmail);
		
		JLabel lblContrasea_1 = new JLabel("contrase\u00F1a :");
		lblContrasea_1.setBounds(56, 243, 87, 20);
		contentPane.add(lblContrasea_1);
		
		textField_usu = new JTextField();
		textField_usu.setBounds(167, 71, 306, 26);
		contentPane.add(textField_usu);
		textField_usu.setColumns(10);
		
		textField_3 = new JTextField();
		textField_3.setBounds(167, 312, 306, 26);
		contentPane.add(textField_3);
		textField_3.setColumns(10);
		
		passwordField = new JPasswordField();
		passwordField.setBounds(167, 153, 306, 26);
		contentPane.add(passwordField);
		
		passwordField_1 = new JPasswordField();
		passwordField_1.setBounds(167, 240, 306, 26);
		contentPane.add(passwordField_1);
		
		JRadioButton rdbtnRecibirNotificaciones = new JRadioButton("Recibir Notificaciones");
		rdbtnRecibirNotificaciones.setBounds(84, 366, 191, 29);
		contentPane.add(rdbtnRecibirNotificaciones);
		
		JRadioButton rdbtnAceptoTerminosY = new JRadioButton("Acepto terminos y politica");
		rdbtnAceptoTerminosY.setBounds(318, 366, 225, 29);
		contentPane.add(rdbtnAceptoTerminosY);
		
		JButton btnRegistrarse = new JButton("Registrarse");
		btnRegistrarse.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				if(rdbtnAceptoTerminosY.isSelected()){
					if(passwordField.getText().equals(passwordField_1.getText())){
					Usuario usu = new Usuario(textField_usu.getText(), passwordField.getText());
					Db.CrearUsuario(usu,url);
					}else{
						JOptionPane.showMessageDialog(null, "Las contraseñas no coinciden");
					}
				}else{
					JOptionPane.showMessageDialog(null, "Acepta los terminos y politica");
				}
			}
		});
		btnRegistrarse.setBounds(309, 407, 115, 29);
		contentPane.add(btnRegistrarse);
		
		JButton btnSalir = new JButton("Salir");
		btnSalir.setBounds(111, 407, 115, 29);
		contentPane.add(btnSalir);
	}
}
