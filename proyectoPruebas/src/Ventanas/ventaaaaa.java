package Ventanas;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.ListSelectionModel;
import javax.swing.border.EmptyBorder;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

import BaseDatos.Db;
import Scraper.LlamadaNoticias;
import ahorcado.ventanas.VentanaJuego;
import datos.Noticia;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.Window;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.BoxLayout;
import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.ScrollPaneConstants;
import java.awt.Font;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

public class ventaaaaa extends JFrame {
	LlamadaNoticias na = new LlamadaNoticias();
	JList<String> lstPosiblesPilotos;
	private String urli = System.getProperty("user.dir")+"/sample1.db";
	
	public ventaaaaa() {
		
		try {
			na.noticasEco();
			na.noticasUlt();
			na.noticiasDepor();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		setTitle("DeustoNews");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 660, 444);
		JPanel contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		contentPane.setLayout(new BorderLayout(0, 0));
		setContentPane(contentPane);
		
		JPanel pnlIndicaciones = new JPanel();
		contentPane.add(pnlIndicaciones, BorderLayout.NORTH);
		pnlIndicaciones.setLayout(new GridLayout(0, 5, 0, 0));
		
		JButton btnUltimasNoticias = new JButton("Ult. Noticias");
		btnUltimasNoticias.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				insertarNoticiasUlt();
			}
		});
		pnlIndicaciones.add(btnUltimasNoticias);
		
		JButton btnEconomia = new JButton("Economia");
		btnEconomia.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent arg0) {
				insertarNoticiasEcono();
			}
		});
		pnlIndicaciones.add(btnEconomia);
		
		JButton btnDeportes = new JButton("Deportes");
		btnDeportes.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				insertarNoticasDepor();
			}
		});
		pnlIndicaciones.add(btnDeportes);
		
		JButton btnPasatiempos = new JButton("Pasatiempos");
		pnlIndicaciones.add(btnPasatiempos);
		btnPasatiempos.addMouseListener(new MouseAdapter(){
			@Override
			public void mouseClicked(MouseEvent arg0){
				VentanaJuego vjg= new VentanaJuego();
				vjg.main(null);
			}
		});
		pnlIndicaciones.add(btnPasatiempos);
		
		JButton btnLogin = new JButton("Login");
		btnLogin.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent arg0) {
				vLoginn log = new vLoginn();
				log.setVisible(true);
			}
		});
		pnlIndicaciones.add(btnLogin);
		
		JPanel pnlContenido = new JPanel();
		contentPane.add(pnlContenido, BorderLayout.CENTER);
		pnlContenido.setLayout(null);
		
		JPanel pnlIzquierda = new JPanel();
		pnlIzquierda.setBounds(0, 0, 356, 339);
		pnlContenido.add(pnlIzquierda);
		pnlIzquierda.setLayout(new BorderLayout(0, 0));
		
		JScrollPane scrlPnlListaIzquierda = new JScrollPane();
		pnlIzquierda.add(scrlPnlListaIzquierda);
		
		lstPosiblesPilotos = new JList<String>();
		lstPosiblesPilotos.setFont(new Font("Tw Cen MT Condensed", Font.PLAIN, 16));
		lstPosiblesPilotos.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		lstPosiblesPilotos.setModel(new DefaultListModel<String>());
		scrlPnlListaIzquierda.setViewportView(lstPosiblesPilotos);
		
		JButton btnVerNoticia = new JButton("Noticia Depor");
		btnVerNoticia.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				NoticiaCompleta com = new NoticiaCompleta(abrirNoticiaDepor(), 1);
				com.setVisible(true);
				
			}
		});
		btnVerNoticia.setBounds(415, 42, 145, 29);
		pnlContenido.add(btnVerNoticia);
		
		JButton btnNoticiaEco = new JButton("Noticia Eco");
		btnNoticiaEco.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent arg0) {

				NoticiaCompleta com = new NoticiaCompleta(abrirNoticiasEco(), 2);
				com.setVisible(true);
			}
		});
		btnNoticiaEco.setBounds(415, 216, 145, 29);
		pnlContenido.add(btnNoticiaEco);
		
		JButton btnUltNoticias = new JButton("Ult. Noticias");
		btnUltNoticias.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				NoticiaCompleta com = new NoticiaCompleta(abrirNoticiasUlt(), 3);
				com.setVisible(true);
			}
		});
		btnUltNoticias.setBounds(415, 131, 145, 29);
		pnlContenido.add(btnUltNoticias);
		
		
		
		
		
		
	}

	public void insertarNoticiasEcono(){
		 DefaultListModel<String> model = (DefaultListModel<String>) lstPosiblesPilotos.getModel();
		 model.removeAllElements();
		 
		
		 for (Noticia piloto : Db.extraerNoticia(urli, "Economia")) {
			String tit = piloto.getTitulo();
			
			if(tit.length() > 63) {
				tit = tit.substring(0, 63) + "...";
			}
			model.addElement(tit);
		}
			
		
	}
	public void insertarNoticasDepor(){
		
		DefaultListModel<String> model = (DefaultListModel<String>) lstPosiblesPilotos.getModel();
		model.removeAllElements();
		
		for (Noticia piloto : Db.extraerNoticia(urli, "Deportes")) {
			String tit = piloto.getTitulo();
			
			if(tit.length() > 63) {
				tit = tit.substring(0, 63) + "...";
			}
			model.addElement(tit);
		}
	
	}
	public void insertarNoticiasUlt() {
		
		 DefaultListModel<String> model = (DefaultListModel<String>) lstPosiblesPilotos.getModel();
		 model.removeAllElements();
		
		 for (Noticia piloto : Db.extraerNoticia(urli, "ultimas")) {
				String tit = piloto.getTitulo();
				
				if(tit.length() > 63) {
					tit = tit.substring(0, 63) + "...";
				}
				model.addElement(tit);
			}
		 }
	public String[] abrirNoticiaDepor(){
		String[] url = new String[10];
		
		url[8]=Db.extraerNoticia(urli, "Deportes").get(lstPosiblesPilotos.getSelectedIndex()).getLink();
		url[9]=Db.extraerNoticia(urli, "Deportes").get(lstPosiblesPilotos.getSelectedIndex()).getTitulo();
		
		return url;
		
	
		
		
	}
	public String[] abrirNoticiasEco(){
		String[] url = new String[10];
		
		url[9]= Db.extraerNoticia(urli, "Economia").get(lstPosiblesPilotos.getSelectedIndex()).getTitulo();
		url[8]= Db.extraerNoticia(urli, "Economia").get(lstPosiblesPilotos.getSelectedIndex()).getLink();
		
		return url;
		
	}
	public String[] abrirNoticiasUlt(){
		String[] url = new String[10];
		
		url[9]= Db.extraerNoticia(urli, "ultimas").get(lstPosiblesPilotos.getSelectedIndex()).getTitulo();
		url[8]= Db.extraerNoticia(urli, "ultimas").get(lstPosiblesPilotos.getSelectedIndex()).getLink();
		
		return url;
		
	}
	
	

	
	public static void main(String[] args){
		ventaaaaa ve = new ventaaaaa();
		ve.setVisible(true);
		


		
	}
}
