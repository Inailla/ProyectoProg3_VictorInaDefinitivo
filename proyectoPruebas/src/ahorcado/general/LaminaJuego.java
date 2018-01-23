package ahorcado.general;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.Rectangle2D;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;

import ahorcado.ventanas.VentanaJuego;
import ahorcado.ventanas.VentanaLoss;
import ahorcado.ventanas.VentanaWin;

public class LaminaJuego extends JPanel { // Clase donde programo lo que va en el interior de la Ventana
	
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 79994572339047373L;
	private Image ahorcadoImagen; // Variable para poder ir mostrando la imagen del ahorcado
	//Componentes swing
	private JButton botonAceptar = new JButton ("Aceptar"); // Boton para aceptar la letra del menu desplegable
	private JLabel etiquetaPalabra = new JLabel(); // Etiqueta que va mostrando la palabra que hay que adivinar
	private JLabel etiquetaAciertos = new JLabel(); // Etiqueta que muestra las letras que vas diciendo
	private JLabel etiquetaMensajes = new JLabel(""); // Etiqueta que muestra vidas
	private JComboBox<String> letras = new JComboBox<String>(); // Menu desplegable donde estan todas las letras
	private JMenuBar menuBarra = new JMenuBar(); // Barra de menu que es la que aparece arriba de todo en los programas
	private JMenu menu = new JMenu ("Menu"); // Lo que va dentro de las barra de menu
	private JMenuItem menuReiniciar = new JMenuItem("Reiniciar"); // Item de la pestaña Menu
	private JMenuItem menuSalir = new JMenuItem("Salir"); // Otro item
	
	private DatosJuego juego = new DatosJuego(); // Inicializo las dos clases para luego usar sus metodos y variables
	private Jugador user = new Jugador("Player 1"); 
	
	public LaminaJuego (){ // Al inicializar esta clase sucede esto:
		try { // Intenta leer esta imagen
			ahorcadoImagen = ImageIO.read(new File ("src/ahorcado/general/Ahorcado6vidas.jpg")); // Uso la variable de tipo Image
		}catch(IOException exception){ 
			System.out.println("La imagen no existe ");
		}
		
		botonAceptar.addActionListener(new EventoAceptarLetra()); // Pongo el boton aceptar a la escucha de la clase que desencadena el evento
		add(botonAceptar); // Agrego el boton a la lamina
		
		etiquetaPalabra.setText(String.valueOf(juego.getLetrasCorrectas())); //Seteo en la etiqueta la variable de la clase DatosJuegos
		
		add(etiquetaPalabra);
		add(etiquetaAciertos);
		add(etiquetaMensajes);
		
		letras.addItem("A");letras.addItem("B");letras.addItem("C");letras.addItem("D");letras.addItem("E");letras.addItem("F");
		letras.addItem("G");letras.addItem("H");letras.addItem("I");letras.addItem("J");letras.addItem("K");letras.addItem("L");
		letras.addItem("M");letras.addItem("N");letras.addItem("O");letras.addItem("P");letras.addItem("Q");letras.addItem("R");
		letras.addItem("S");letras.addItem("T");letras.addItem("U");letras.addItem("V");letras.addItem("W");letras.addItem("X");
		letras.addItem("Y");letras.addItem("Z"); //Agrego los items al combobox
		add(letras); // Agrego a la lamina el combobox
		
		menu.add(menuReiniciar); // Al boton menu les agrego los botones salir y agregar
		menu.add(menuSalir); 
		menuBarra.add(menu); // A la barra de menus le agrego el boton menu donde adentro estan salir y reiniciar
		add(menuBarra); // agrego a la lamina la barra
		
		menuReiniciar.addActionListener(new EventoReiniciar());
		menuSalir.addActionListener(new EventoSalir());  
		
		
	}
	
	public void paintComponent (Graphics g){ // Metodo que sirve para dibujar,controlar colores,fuentes,ubicar botones etiquetas, etc
		
		super.paintComponent(g);
		
		Graphics2D g2 = (Graphics2D)g; // uso el Graphics2D para construir figuras
		
		Rectangle2D rectanguloPalabra = new Rectangle2D.Double(1, 1, 150, 75); //Construyo un rectangulo y le paso por parametro el tamaño
		Rectangle2D rectanguloAciertos = new Rectangle2D.Double(1, 82, 150, 75);// y la localizacion
		Rectangle2D rectanguloLetra = new Rectangle2D.Double(1, 163, 150, 75); // cada rectangulo para cada grupo y sus funcionalidades
		Rectangle2D rectanguloMensaje = new Rectangle2D.Double(1, 243, 392, 75);

		g2.setFont(new Font("Arial",Font.BOLD,13)); // seteo la fuente, estilo y tamaño para hacer lo siguiente:
		// Asigno color con el fill y pinto el fondo y en el drawString escribo con la fuente que elegi antes
		g2.setPaint(new Color(50,130,140));g2.fill(rectanguloPalabra);g2.setColor(Color.BLACK);g.drawString("Palabra: ",3,15);
		g2.setPaint(new Color(50,130,140));g2.fill(rectanguloAciertos);g2.setColor(Color.BLACK);g.drawString("Aciertos: ",3,95);
		g2.setPaint(new Color(50,130,140));g2.fill(rectanguloLetra);g2.setColor(Color.BLACK);g.drawString("Letra: ",3,175);
		g2.setPaint(new Color(50,130,140));g2.fill(rectanguloMensaje);g2.setColor(Color.BLACK);g.drawString("Mensaje: ",3,255);
		
		g.drawImage(ahorcadoImagen,160,5,null); // Muestro la imagen del ahorcado
		
		botonAceptar.setLocation(68,190); // Ubico a los componentes Swing en la lamina o fondo de la ventana, por parametros van los ejes x y
		
		etiquetaPalabra.setLocation(5,35);
		etiquetaAciertos.setLocation(5,115);
		etiquetaMensajes.setLocation(5,270);
		
		letras.setLocation(10, 190);
		
		menuBarra.setLocation(320,0);
		
	}
	
	private class EventoAceptarLetra implements ActionListener {//Clase privada que uso para deseencadenar acciones en eventos
		
		int correctos=0; // Cantidad descubiertas de la palabra secreta
		boolean incorrecto=true;// Variable para descontar una vida
		
		Mensaje msj = new Mensaje(); //Instancio la clase mensaje para ir mostrando mensajes y vidas

		public void actionPerformed(ActionEvent e) { 
			
			Object botonPulsado = e.getSource(); 
			String aciertos=(String)letras.getSelectedItem()+"-"; // Variable para ir mostrando los aciertos
			
			char letraPosicion []= new char[juego.getLetrasCorrectas().length]; //Guardo las letras para compararlas
			letraPosicion = juego.getLetrasCorrectas(); // charArray para mostrar lo que hay que adivinar
			
			if (botonPulsado==botonAceptar){ 
				
				etiquetaAciertos.setText(etiquetaAciertos.getText()+aciertos); //Esta etiqueta va acumulando las letras que se ingresan
				
				for(int i=0;i<juego.getPalabraSecreta().toString().length();i++){ // for para comprarar si hay coincidencia en las letras
				// Se debe transformar en String y usar el charAt 0 para que funcione
				if(letras.getSelectedItem().toString().charAt(0)==juego.getPalabraSecreta().toString().charAt(i)){
					letraPosicion[i*2]=letras.getSelectedItem().toString().charAt(0); //El *2 porque hay espacios en blanco
					etiquetaPalabra.setText(String.valueOf(letraPosicion)); // Voy cambiando la etiqueta al ir descubriendo las letras
					incorrecto=false; // Evito que me descuente vidas al acertar alguna letra
					correctos++; // Cuento la cantidad de aciertos y luego los comparo con el total de la palabra secreta para ver si el
					// usuario ganó
					
				}
				}
				
				letras.removeItemAt(letras.getSelectedIndex()); // borro del combobox la letra ingresada
				
				if(incorrecto){ // Si incorrecto es verdadero se pierde una vida
					user.pierdeVida();
				}incorrecto=true; // Restablezco a true para que no me descuente otra vida 
				
				// Va cambiando la imagen y los mensajes segun las vidas
				if(user.getVidas()==6){ 
					msj.muchasVidas();
					etiquetaMensajes.setText(msj.getMensaje()+"(Vidas:"+user.getVidas()+")");
				}else if(user.getVidas()==5){
					msj.muchasVidas();
					etiquetaMensajes.setText(msj.getMensaje()+"(Vidas:"+user.getVidas()+")");
					try { 
						ahorcadoImagen = ImageIO.read(new File ("src/ahorcado/general/Ahorcado5vidas.jpg"));
					}catch(IOException exception){
						System.out.println("La imagen no existe ");
					}
				}else if (user.getVidas()==4){
					msj.mitadVidas();
					etiquetaMensajes.setText(msj.getMensaje()+"(Vidas:"+user.getVidas()+")");
					try { 
						ahorcadoImagen = ImageIO.read(new File ("src/ahorcado/general/Ahorcado4vidas.jpg"));
					}catch(IOException exception){
						System.out.println("La imagen no existe ");
					}
				}else if(user.getVidas()==3){
					etiquetaMensajes.setText(msj.getMensaje()+"(Vidas:"+user.getVidas()+")");
					try { 
						ahorcadoImagen = ImageIO.read(new File ("src/ahorcado/general/Ahorcado3vidas.jpg"));
					}catch(IOException exception){
						System.out.println("La imagen no existe ");
					}
				}else if (user.getVidas()==2){
					msj.pocasVidas();
					etiquetaMensajes.setText(msj.getMensaje()+"(Vidas:"+user.getVidas()+")");
					try { 
						ahorcadoImagen = ImageIO.read(new File ("src/ahorcado/general/Ahorcado2vidas.jpg"));
					}catch(IOException exception){
						System.out.println("La imagen no existe ");
					}
				}else if(user.getVidas()==1){
					etiquetaMensajes.setText(msj.getMensaje()+"(Vidas:"+user.getVidas()+")");
					try { 
						ahorcadoImagen = ImageIO.read(new File ("src/ahorcado/general/Ahorcado1vida.jpg"));
					}catch(IOException exception){
						System.out.println("La imagen no existe ");
					}
				}else if(user.getVidas()==0){ // Si el usuario pierde se deshabilitan el boton y el combobox
					msj.perdistes();
					etiquetaMensajes.setText(msj.getMensaje());
					botonAceptar.setEnabled(false);
					try { 
						ahorcadoImagen = ImageIO.read(new File ("src/ahorcado/general/Ahorcado0vidas.jpg"));
					}catch(IOException exception){
						System.out.println("La imagen no existe ");
					}
					
					VentanaLoss loss = new VentanaLoss(); // Tambien aparece la ventana de que perdistes
					
				}
				
				if(correctos==juego.getPalabraSecreta().toString().length()){ 
					msj.ganastes();
					etiquetaMensajes.setText(msj.getMensaje());
					botonAceptar.setEnabled(false);
					
					VentanaWin win = new VentanaWin(); // Lo mismo que al perder al ganar
					
				}
		}
	}
	}
	
	private class EventoReiniciar implements ActionListener{ //Al tocar en reiniciar inicio el programa de nuevo

		public void actionPerformed(ActionEvent e) {
			
			VentanaJuego v = new VentanaJuego();
		}
	}
	
	private class EventoSalir implements ActionListener{ // Si se toca en salir se cierra la ventana y deja de compilar

		public void actionPerformed(ActionEvent e) {
			
			System.exit(-1);
			
		}
	}
}