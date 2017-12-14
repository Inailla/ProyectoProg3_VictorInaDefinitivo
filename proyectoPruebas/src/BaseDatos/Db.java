package BaseDatos;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import Scraper.LlamadaNoticias;
import datos.Usuario;



public class Db {
	
	static LlamadaNoticias oli = new LlamadaNoticias();
	static Statement statement;
	static String noticia,link;
	static String nombre,pass;
	
    public static String insertarNoticiasDeporBd() throws IOException{
		
		for (int i = 1; i < oli.noticiasDepor().size(); i++) {
			noticia=oli.noticiasDepor().get(i).titulo;
			link=oli.noticiasDepor().get(i).link;
			try {
				statement.executeQuery("insert into noticiaDEP values('"+noticia+"','"+link+"')");
				
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			 try {
				ResultSet rp = statement.executeQuery("select * from noticiaECO");
				while(rp.next()){
					return rp.getString("titulo");
				}
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return null;
		
		
	}
    public static String insertarNoticiasEcoBd() throws IOException{
    	for (int ij = 1; ij < oli.noticasEco().size(); ij++) {
			noticia=oli.noticasEco().get(ij).titulo;
			link=oli.noticasEco().get(ij).link;
			try {
				statement.executeQuery("insert into noticiaECO values('"+noticia+"','"+link+"')");
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
    	  try {
			ResultSet rs = statement.executeQuery("select * from noticiaDEP");
			while(rs.next()){
				return rs.getString("titulo");
			}
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
    }
    public static void comprobarUsuario(Usuario us){
    	nombre=us.getUser();
    	pass=us.getPass();
    	try {
			
			ResultSet rb = statement.executeQuery("selecect * from usuario where usuario = '"+nombre+"' and pass = '"+pass+"'");
			while(rb.next()){
				
			}
			
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("error usuario");
		}
    }

	public static void main(String[] args) throws ClassNotFoundException, IOException

	  {

	    // Carga el sqlite-JDBC driver usando el cargador de clases

	    Class.forName("org.sqlite.JDBC");
	    



	    Connection connection = null;

	    try

	    {

	      // Crear una conexi�n de BD

	      connection = DriverManager.getConnection("jdbc:sqlite:sample.db");

	      statement = connection.createStatement();

	      statement.setQueryTimeout(30);  // poner timeout 30 msg



	    
	      statement.executeUpdate("drop table if exists noticiaECO");
	      statement.executeUpdate("drop table if exists noticiaDEP");
	      statement.executeUpdate("drop table if exists usuario");

	      statement.executeUpdate("create table noticiaECO (titulo string, link string)");
	      statement.executeUpdate("create table noticiaDEP (titulo string, link string)");
	      statement.executeUpdate("create table usuario (usario string, pass string)");
	      
	      statement.executeQuery("insert into noticiaECO values('ignacio','1234por')");
	      statement.executeQuery("insert into noticiaECO values('pablo','telepizza')");
	      
	      insertarNoticiasEcoBd();
          insertarNoticiasDeporBd();
	     

	      
	      ResultSet rp = statement.executeQuery("select * from noticiaECO");
	      

	      while(rp.next())

	      {

	        // Leer el resultset

	       
	        
	        System.out.println(rp.getString("link"));

	      }

	    } catch(SQLException e) {

	      System.err.println(e.getMessage());

	    } finally {

	      try

	      {

	        if(connection != null)

	          connection.close();

	      }

	      catch(SQLException e)

	      {

	        // Cierre de conexi�n fallido

	        System.err.println(e);

	      }

	    }

	  }
}