package java_practice;

public class Base_7 {

	public static void main(String[] args) {
		int num=-8;
		String ans=convertToBase7(num);
		System.out.print(ans);
		

	}
    public static String convertToBase7(int num) {
    	StringBuffer sb = new StringBuffer();
    	while(num/7 !=0){
    		sb.insert(0, Math.abs(num%7));
    		num=num/7;
    	}
    	sb.insert(0,num%7 );
    	return sb.toString();
    }

}
