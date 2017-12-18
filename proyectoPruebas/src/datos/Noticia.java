package datos;

import java.text.SimpleDateFormat;
import java.util.Date;

public class Noticia {

	public String titulo;
	public String link;
	public String categoria;
	public String fecha;
	
	public Noticia(String titulo, String link,String categoria) {
		this.titulo = titulo;
		this.link = link;
		this.categoria = categoria;
		this.fecha = new SimpleDateFormat("yyyy-MM-dd").format(new Date());
	}
	
	public Noticia(String titulo, String link,String categoria, String fecha) {
		this.titulo = titulo;
		this.link = link;
		this.categoria = categoria;
		this.fecha = fecha;
	}
	
	
	public String getFecha(){
		return fecha;
	}
	public String getCategoria(){
		return categoria;
	}

	public String getTitulo() {
		return titulo;
	}

	public void setTitulo(String titulo) {
		this.titulo = titulo;
	}

	public String getLink() {
		return link;
	}

	public void setLink(String link) {
		this.link = link;
	}
	
	
}
