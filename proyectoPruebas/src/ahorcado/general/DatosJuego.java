package ahorcado.general;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DatosJuego { // Clase para que el juego pueda funcionar
	
	private Palabra palabraSecreta; // Variable de tipo Palabra para guardar la palabra que hay que adivinar
	
	private int digitosPS; // Variable que me almacena los digitos de la palabra secreta
	private char [] letras; // Vector que me guarda las letras de la palabra secreta para luego compararlas una por una
	private char [] letrasCorrectas; // Vector para mostrar lo que hay que adivinar
	
	private Random rnd = new Random (); // Lo uso para elegir una palabra aelatoria entre las 10 que hay
	
	private List <Palabra> palabras = new ArrayList <Palabra>(); // Lista con las palabras posibles
	
	public DatosJuego(){
		palabras.add (new Palabra("CANGURO"));
		palabras.add (new Palabra("COMPUTADORA"));
		palabras.add (new Palabra("MACRI"));
		palabras.add (new Palabra("LEYENDA"));
		palabras.add (new Palabra("MOCHILA"));
		palabras.add (new Palabra("PASTEL"));
		palabras.add (new Palabra("PELOTA"));
		palabras.add (new Palabra("CRISTAL"));
		palabras.add (new Palabra("MAGO"));
		palabras.add (new Palabra("AYUNTAMIENTO"));
		
		palabraSecreta = palabras.get(rnd.nextInt(palabras.size())); // Elijo una palabra al azar

		digitosPS = palabraSecreta.toString().length();// Cuenta los digitos de la palabra que hay que adivinar
		letras = palabraSecreta.toString().toCharArray();// Transforma la palabra que hay que adivinar a String y luego los guarda en un Array
		letrasCorrectas = new char [digitosPS*2];// Defino el tamaño de un array y lo multiplico por dos para que me ponga espacios en blanco
		
		for (int j=0;j<digitosPS*2;j++){// transformo cada letra a un guion bajo y luego le pongo un espacio 
			letrasCorrectas[j]='_';
			j++;
			letrasCorrectas[j]=' ';
		}
	}

	
	public Palabra getPalabraSecreta() {
		return palabraSecreta;
	}

	public char[] getLetrasCorrectas() {
		return letrasCorrectas;
	}

	public char[] getLetras() {
		return letras;
	}
	


}
