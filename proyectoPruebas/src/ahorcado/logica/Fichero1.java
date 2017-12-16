package ahorcado.logica;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.FileSystemNotFoundException;
import java.util.HashMap;

import javax.swing.JOptionPane;

public class Fichero1 {
	public static String usuario1 = "";
	public static String pass1 = "";

	public static HashMap<String, String> Gestion() throws NullPointerException, FileSystemNotFoundException {

		File fichero = new File("..\\git\\ProyectoProg3_VictorInaDefinitivo\\proyectoPruebas\\src\\ahorcado.general\\archivo.txt");
		
		HashMap<String, String> usuariosDelFichero = new HashMap<>();
		try {

			FileReader fr = new FileReader(fichero);
			BufferedReader br = new BufferedReader(fr);

			String linea = br.readLine();
			while ((linea != null)) {

				String[] usuarios = linea.split(";");
				usuariosDelFichero.put(usuarios[0], usuarios[1]);
				linea = br.readLine();

			}
			br.close();
		} catch (Exception e) {
			try {
			} catch (NullPointerException e1) {
				JOptionPane.showMessageDialog(null, " Un valor nulo no es valido");
				// System.out.println("Algo nulo no es valido " + fichero + ": "
				// + e1);
			} catch (FileSystemNotFoundException e2) {
				JOptionPane.showMessageDialog(null, "No se encuentra el archivo");
				// System.out.println("No se encuentra el archivo " + fichero +
				// ": " + e2);
			}
			return usuariosDelFichero;

		}
		return usuariosDelFichero;
	}

}