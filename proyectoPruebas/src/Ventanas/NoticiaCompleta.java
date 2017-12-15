package Ventanas;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

import Scraper.LlamadaNoticias;
import Scraper.TestURLImage;

import javax.swing.JLabel;
import javax.swing.JOptionPane;

import java.awt.Font;
import java.awt.ScrollPane;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import javax.swing.JTextPane;
import javax.swing.JTextArea;
import javax.swing.JTable;
import javax.imageio.ImageIO;
import javax.swing.DropMode;
import javax.swing.ImageIcon;

import java.awt.Scrollbar;
import java.awt.image.BufferedImage;

public class NoticiaCompleta extends JFrame {
	LlamadaNoticias ana = new LlamadaNoticias();
	JTextArea textPane;
	JLabel label_tit;
	ScrollPane bar;
	TestURLImage test;
	JLabel lblimagen;
	JPanel panel_imagen;
	public NoticiaCompleta(String[] strings,int i) {
		
		setTitle("DeustoNews");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setBounds(100, 100, 770, 580);
		getContentPane().setLayout(null);
		
		JPanel panel_titulo = new JPanel();
		panel_titulo.setBounds(15, 16, 718, 87);
		getContentPane().add(panel_titulo);
		
		label_tit = new JLabel(".");
		label_tit.setFont(new Font("Trebuchet MS", Font.PLAIN, 28));
		panel_titulo.add(label_tit);
		
		JPanel panel_texto = new JPanel();
		panel_texto.setBounds(15, 119, 326, 389);
		getContentPane().add(panel_texto);
		
		textPane = new JTextArea(20,40);
		textPane.setLineWrap(true);
		textPane.setWrapStyleWord(true);
		textPane.setEditable(false);
		
		
		
		panel_texto.add(textPane);
		
		panel_imagen = new JPanel();
		panel_imagen.setBounds(379, 119, 354, 389);
		getContentPane().add(panel_imagen);
		
		lblimagen = new JLabel("");
		try {
			insertarImagen(strings[8],i);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		
		try {
			insertarTexto(strings[8],i);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		tituloTamaño(strings[9]);
		
		
		
	}
	public void insertarImagen(String ima,int o) throws IOException{
		if(o==1){
		String urlDeportes = ima;
		Document d = Jsoup.connect(urlDeportes).get();
		Elements el = d.select("div#wrapper");
		for (org.jsoup.nodes.Element element : el.select("div.post-thumbnail")) {
			String imagen = element.select("div.post-thumbnail img").attr("src");
			imagenFrame(imagen);
			}}
		else{
		String urlDeportes = ima;
		Document d = Jsoup.connect(urlDeportes).get();
		Elements el = d.select("div#cuerpo_noticia");
		for (org.jsoup.nodes.Element element : el.select("div.foto-ancho")) {
			String imagen = element.select("div.foto-ancho img").attr("src");
			imagenFrame(imagen);
			}}
		
		
	}
	public void insertarTitulo(String tit){
		label_tit.setText(tit);
	}
	public void tituloTamaño(String p){
		if(p.length()>60){
			p="<html>"+p.substring(0, 60)+"<br>"+p.substring(60, p.length())+"</html>";
			insertarTitulo(p);
		}else{
			insertarTitulo(p);
		}
	}
	public void insertarTexto(String url,int o) throws IOException{//problema aqui
		if(o==1){
		Document d1 = Jsoup.connect(url).get();
		Elements el1 = d1.select("div#wrapper");
		for (org.jsoup.nodes.Element element : el1.select("div.entry-inner")) {
			String full= element.select("div.entry-inner").text();
			textPane.setText(full);
		}}else{//NO LEE
		Document d1 = Jsoup.connect(url).get();
		Elements el1 = d1.select("div#cuerpo_noticia");
		for (org.jsoup.nodes.Element element : el1.select("div.cuerpo_noticia")) {
			System.out.println(element.text());
			textPane.setText(element.text());
			
		}}
		
	}
	public void imagenFrame(String g){
		 try {
             String path = g;
             System.out.println("Get Image from " + path);
             URL url = new URL(path);
             BufferedImage image = ImageIO.read(url);
             System.out.println("Load image into frame...");
             lblimagen = new JLabel(new ImageIcon(image));
             panel_imagen.add(lblimagen);
             
             
            
         } catch (Exception exp) {
             exp.printStackTrace();
         }

	}
}
