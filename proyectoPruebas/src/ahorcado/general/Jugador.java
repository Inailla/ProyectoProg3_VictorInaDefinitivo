package ahorcado.general;

	
public class Jugador {
	
	private String usuario;
	private int vidas=6; // defino las vidas del usuario como decia en la consigna
	
	public Jugador (String usuario){ // al crear un jugador le paso como parametro string nombre de usuario o uso el constructor por defecto
		this.usuario=usuario;
		
	}
	
	public int pierdeVida(){ // metodo para que el usuario pierda vidas al equivocarse
		vidas--;
		return vidas;
	}
	
	
		
	public String getUsuario() {
		return usuario;
	}
	public void setUsuario(String usuario) {
		this.usuario = usuario;
	}
	public int getVidas() {
		return vidas;
	}
	public void setVidas(int vidas) {
		this.vidas = vidas;
	}
	
	
}
