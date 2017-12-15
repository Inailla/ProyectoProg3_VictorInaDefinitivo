package BaseDatos;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import Scraper.LlamadaNoticias;
import datos.Noticia;
import datos.Usuario;



public class Db {
	
	static LlamadaNoticias oli = new LlamadaNoticias();
	static Statement statement;
	static String noticia,link;
	static String nombre,pass;
	
	public static Connection iniDB(String url){
		
		   try {
			Class.forName("org.sqlite.JDBC"); 
			Connection connection =  DriverManager.getConnection("jdbc:sqlite:"+url);
			return connection;
		} catch (ClassNotFoundException | SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
		    
          
}
	public static void CrearTablas(String url){
		Connection con = iniDB(url);
		
		try {
			statement = con.createStatement();
			
			 statement.executeUpdate("create table if not exists noticias (titulo text UNIQUE, link text,categoria text,fecha date)");
		     statement.executeUpdate("create table if not exists usuario (usario text UNIQUE, pass text)");
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}finally {

		      try

		      {

		        if(con != null)

		          con.close();

		      }

		      catch(SQLException e)

		      {

		        // Cierre de conexión fallido

		        System.err.println(e);

		      }

		    }
	}
	public static void insertNews(String url,ArrayList<Noticia> arr,String cat){
		String date= new SimpleDateFormat("yyyy-MM-dd").format(new Date());
		Connection con= iniDB(url);
		String sql = "insert into noticias values (?,?,?,?);";
		PreparedStatement sta = null;
		try {
			sta = con.prepareStatement(sql);
			con.setAutoCommit(false);
			for(Noticia not : arr) {
				sta.setString(1, not.getTitulo());
				sta.setString(2, not.getLink());
				sta.setString(3, cat);
				sta.setDate(4, java.sql.Date.valueOf(date));
				sta.addBatch();
			}
			sta.executeBatch();
			con.commit();
			
		} catch (SQLException e) {
			try{
				con.rollback();
			}catch (Exception e1) {
				// TODO: handle exception
				e1.printStackTrace();
			}
		
			}finally {

			      try

			      {

			        if(con != null)

			          con.close();

			      }

			      catch(SQLException e)

			      {

			        // Cierre de conexión fallido

			        System.err.println(e);

			      }
			      }

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

	
	  
}
