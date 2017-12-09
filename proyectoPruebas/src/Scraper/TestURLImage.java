package Scraper;

import java.awt.EventQueue;
import java.awt.image.BufferedImage;
import java.net.URL;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

public class TestURLImage {


public static void main(String[] args) {
    new TestURLImage();
}

public TestURLImage() {
    EventQueue.invokeLater(new Runnable() {
        @Override
        public void run() {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch( ClassNotFoundException |IllegalAccessException |InstantiationException | UnsupportedLookAndFeelException ex) {
            }

            try {
                String path = "https://www.losotros18.com/wp-content/uploads/2017/11/640x360_17224636gir-rsoc0943-520x245.jpg";
                System.out.println("Get Image from " + path);
                URL url = new URL(path);
                BufferedImage image = ImageIO.read(url);
                System.out.println("Load image into frame...");
                JLabel label = new JLabel(new ImageIcon(image));
                JFrame f = new JFrame();
                f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                f.getContentPane().add(label);
                f.pack();
                f.setLocation(200, 200);
                f.setVisible(true);
            } catch (Exception exp) {
                exp.printStackTrace();
            }

        }
    });
}
}