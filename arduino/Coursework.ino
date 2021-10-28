#include <Wire.h>
#include <Adafruit_RGBLCDShield.h>
#include <utility/Adafruit_MCP23017.h>

Adafruit_RGBLCDShield lcd = Adafruit_RGBLCDShield();


void setup() {
  lcd.begin(16, 2);
  Serial.begin(9600);

    // main menu
  Serial.println("SEL to Practice");
  lcd.print("SEL to Practice");
  lcd.setCursor(0, 1);
  Serial.println("DOWN for Story");
  lcd.print("DOWN for Story");
  randomSeed(analogRead(0)); // random seed
}



int D = 500; // delay between seeing the different buttons
// buttons are displayed on screen for 500 more ms so repeated buttons are clear
// hence incrementing by 1000 goes from 1 second to two seconds
double sec = ((D+500) / 1000); // converts the value of D into seconds

int points = 0;
int level = 1;

// variables to monitor button presses
boolean s_pressed = false;
boolean u_pressed = false;
boolean d_pressed = false;
boolean l_pressed = false;
boolean r_pressed = false;
boolean story = false;

int diff=2; // integer used for the difficulty 1 is easy, 2 is medium, 3 is hard
int *S = NULL; // pointer to array that will contain the sequence
int *A = NULL ; // pointer to array that will contain the user's answers
int answer_tracker = 0; // variable used to compare the index of array containing the user's answers to the
// array containing the correct answers
int N= 4; // default number of answers is 4
int *M = NULL; // pointer to array that will contain the possible buttons
int M_size = 2; // contains how big the set M will be.


// variables controlling state transitions
boolean shown = false; // used to control how long text should be displayed for
boolean wrong = false; // turns true if the user has entered the wrong sequence.
boolean change = false; // controls a state change in loop()


// state for changing the size of M
void change_M() {
  lcd.setCursor(0, 0);
  if (shown == false & story == false) {
    lcd.clear();
    Serial.println("NUMBER OF POSSIBLE BUTTONS");
    lcd.print("NUMBER OF");
    lcd.setCursor(0,1);
    lcd.print("POSSIBLE BUTTONS");
    lcd.setCursor(0,0);
    shown = true;
  }
  delay(1000);
  lcd.clear();
  while (true) {
    lcd.setCursor(0, 0);
    lcd.print(M_size);
    // allows user to use left and right buttons in order
    // to change the values
    int buttons = lcd.readButtons();
    if (buttons & BUTTON_LEFT) { 
      l_pressed = true;   
    } else if ( l_pressed & M_size > 2) {
      M_size = M_size - 1;
      lcd.clear();
      lcd.print(M_size);
      l_pressed = false;
    }

    if (buttons & BUTTON_RIGHT) { 
      r_pressed = true;   
    } else if (r_pressed & M_size < 4) {
      M_size = M_size + 1;
      lcd.clear();
      Serial.println(M_size);
      lcd.print(M_size);
      r_pressed = false;
    }

    // Write's M_size number of characters to M
    if (buttons & BUTTON_DOWN) { 
      d_pressed = true;   
    } else if (d_pressed) {
        M = (int*)malloc(M_size * sizeof(int));
        for (int i = 0; i < (M_size) ; i) {
            M[i] = random(1,5);
            if (M[i]!=M[i-1] || i==0){
              i=i+1;
            }
        }
        d_pressed=false;
      Serial.println("Done");
      d_pressed = false;
      s_pressed = false;
      change = true;
      shown = false;
      if (story) {
        loop();
      } else {
        change_D();
      }
    }
  }  
}


// state for the results and freeing up memory
void results() {

  //free memory for the arrays
  free(A);
  free(S);
  shown = false;
  answer_tracker = 0;
  // displays appropriate message  and flashes the screen red
  // if user entered an incorrect answer
  if (wrong) {
    lcd.setBacklight(1);
    Serial.println("failiure");
    lcd.print("failiure");
    delay(2000);
    change = false;
    wrong = false;
    s_pressed = false;
    d_pressed = false;
    story = false;
    // displays story mode results
    if (story) {
      lcd.clear();
      Serial.println("Points:");
      lcd.print("Points:");
      Serial.println(points);
      lcd.print(points);
      delay(2000);

      // resets the values
      diff=2;
      D = 3500;
      points = 0;
      level = 1;
      M_size = 2;
      N= 4;
    }
    delay(1000);
    lcd.setBacklight(-1);
    lcd.clear();

    // message for main menu before transitioning
    // back to main menu state
    Serial.println("SEL for Practice");
    lcd.print("SEL for Practice");
    Serial.println("DOWN for Story");
    lcd.setCursor(0, 1);
    lcd.print("DOWN for Story");
    loop();
   } 
    // displays appropriate message and flashes the screen green
    // if the user entered
    // all of the answers correctly 
    else {
    lcd.setBacklight(2);
    Serial.println("Success!");
    lcd.print("Success!");
    delay(2000);
    lcd.clear();
    if (story == false) {
      Serial.println("SEL to Practice");
      lcd.print("SEL to Practice");
      lcd.setCursor(0, 1);
      Serial.println("DOWN for Story");
      lcd.print("DOWN for Story");
      
      change = false;
      wrong = false;
      s_pressed = false;
      d_pressed = false;
      loop();
    }
    if (story) {
      level += 1;
      Serial.println("Points:");
      lcd.print("Points:");
      Serial.println(points);
      lcd.print(points);
      delay(2000);
      lcd.setBacklight(-1);
      story_setup();
    }
    change_M();
  }
}

// state for the actual game
void game_loop() {

  while (true) {
    int buttons = lcd.readButtons();
    
    if (shown == false) {
      lcd.clear();
      Serial.println("REPEAT THIS");
      lcd.print("REPEAT THIS");
      lcd.setCursor(0,1);
      lcd.print("SEQUENCE");
      delay(500); 
      lcd.setCursor(0,0);
      // shows the sequence on the serial and lcd
      for (int i = 0; i <= (N); i++) {
        lcd.clear();
        delay(500); // here to make it more clear if there is a repeated button in the sequence.
        if (S[i] == 1) {
          Serial.println("LEFT");
          lcd.print("LEFT");
        }
        if (S[i] == 2) {
          Serial.println("RIGHT");
          lcd.print("RIGHT");
        }
        if (S[i] == 3) {
          Serial.println("UP");
          lcd.print("UP");
        }
        if (S[i] == 4) {
          Serial.println("DOWN");
          lcd.print("DOWN");
        }
        delay(D);
        if (i==N){
          delay(500);
        }
        lcd.clear();
      } shown = true;
    }

    // handles correct and incorrect button presses
    // by using answer_tracker to compare the array 
    // that stores the correct answers to the array 
    // that stores the user's answers
    if (buttons & BUTTON_LEFT) { 
      l_pressed = true;   
    } else if ( l_pressed) {
      A[answer_tracker] = 1;
      if (A[answer_tracker] == 1 & S[answer_tracker] == 1 & l_pressed) {

        Serial.println("Correct");
        points += 1;
        
      } else {
        wrong = true; // prompts state transition into game over
      }
      answer_tracker += 1;
      l_pressed = false;

    }

  
    if (buttons & BUTTON_RIGHT) { 
      r_pressed = true;   
    } else if ( r_pressed) {
      A[answer_tracker] = 2;
      if (A[answer_tracker] == 2 & S[answer_tracker] == 2 & r_pressed) {

        Serial.println("Correct");
        points += 1;
        
      } else {
        wrong = true;
      }
      answer_tracker += 1;
      r_pressed = false;
    }

    if (buttons & BUTTON_UP) { 
      u_pressed = true;   
    } else if ( u_pressed) {
      A[answer_tracker] = 3;
      if (A[answer_tracker] == 3 & S[answer_tracker] == 3 & u_pressed) {
        Serial.println("Correct");
        points += 1;
        
      } else {
        wrong = true;
      }
      answer_tracker += 1;
      u_pressed = false;
    }


    if (buttons & BUTTON_DOWN) { 
      d_pressed = true;   
    } else if ( d_pressed) {
      A[answer_tracker] = 4;
      if (A[answer_tracker] == 4 & S[answer_tracker] == 4 & d_pressed) {
        Serial.println("Correct");
        points += 1;
      } else {
        wrong = true;
      }
      answer_tracker += 1;
      d_pressed = false;
    }

    if (answer_tracker == (N)) {
      results(); // state transition into displaying results
    }
  }
}

// state for changing the difficulty
void difficulty(){
  lcd.setCursor(0,0);
  if (shown == false) {
    lcd.clear();
    Serial.println("ALTER DIFFICULTY WITH LEFT AND RIGHT");
    lcd.print("ALTER DIFFICULTY");
    lcd.setCursor(0,1);
    lcd.print("WITH LEFT&RIGHT");
    lcd.setCursor(0,0);
    delay(2000);
    lcd.clear();
    shown = true;
  }
  while (true) {
    shown=false;
    int buttons = lcd.readButtons();
    // shows current difficulty on LCD screen

    if (diff==3){
      lcd.print("HARD");
    }
    if (diff==2){
      lcd.print("MEDIUM");
    }
    if (diff==1){
      lcd.print("EASY");
    }

    // use left and right buttons to change the difficulty
    if (buttons & BUTTON_LEFT) { 
        l_pressed = true;   
      } else if ( l_pressed & diff>1) {
        lcd.clear();
        diff= diff- 1;
        Serial.println(diff);
        l_pressed = false;
      }
  
      if (buttons & BUTTON_RIGHT) { 
        r_pressed = true;   
      } else if (r_pressed & diff<3) {
        lcd.clear();
        diff= diff+ 1;
        Serial.println(diff);
        r_pressed = false;
      }
      
      if (buttons & BUTTON_DOWN) { 
      d_pressed = true;   
      } else if (d_pressed) {
        d_pressed = false;
        story_setup(); // state transition into the story mode
      }
    delay(100);
    lcd.clear();
  }
}
// method for the story mode
void story_setup() {
  while (true) {
    lcd.setBacklight(-1);
    lcd.clear();
    Serial.println("LEVEL");
    lcd.print("LEVEL: ");
    Serial.println(level);
    lcd.print(level);
    delay(2000);
    lcd.clear();
    if (level==1){
      // resets the values for level 1 
      // in case the user changed them in practice mode
      diff==2;
      D=3500; // sets the inital value of D to 4 seconds
      points = 0;
      level = 1;
      M_size = 2;
      N = 4;
    }

   

    story = true;
    change = true;

    // settings for each difficulty
    // gets faster and longer every fourth level
    if (diff==1 & level % 4 == 0 ){
      N+=1;
      if(D>500){
        D-=1000;
      }
      if (M_size<4){
        M_size+=1;
      }
    }
    
    // gets faster and longer every third level
    if (diff==2 & level % 3 == 0 ){
      N+=1;
      if(D>500){
        D-=1000;
      }
      if (M_size<4){
        M_size+=1;
      }
    }

    // gets faster and longer every other level
    if (diff==3 & level % 2 == 0 ){
      N+=1;
      if(D>500){
        D-=1000;
      }
      if (M_size<4){
        M_size+=1;
      }
    }
  
  
 
    // makes all four buttons possible in the sequence after a certain level
    if (M_size==4){
      for (int i =1; i<(M_size); i++){
        M[i-1]=i;
        Serial.println(M[i-1]);
      } 
    } else{    // allocates memory and writes to an array containing some of the possible buttons up to that point
        M = (int*)malloc(M_size * sizeof(int));
        for (int i = 0; i < (M_size) ; i) {
            M[i] = random(1,5);
            Serial.println(M[i]);
            if (M[i]!=M[i-1] || i==0){
              i=i+1;
            }
        }
    }

    // alloctes memory for the array containing the 
    // sequence and the array containing the user's answers
    S = (int*)malloc(N * sizeof(int));
    A = (int*)malloc(N * sizeof(int));
    for (int i = 0; i <= (N-1) ; i++) {
      S[i] = M[random(0,M_size)];
    }
    game_loop(); // state transition into game loop

  }
}

// method for changing the delay on screen
void change_D() {
  lcd.setCursor(0, 0);

  if (shown == false & story == false) {
    lcd.clear();
    Serial.println("CHANGE DELAY");
    lcd.print("CHANGE DELAY");
    shown = true;
    delay(1000);
    lcd.clear();
  }
  while (true) {
    lcd.setCursor(0, 0);
    int buttons = lcd.readButtons();
    
    lcd.print("Seconds:");
    lcd.setCursor(0, 1);
    lcd.print(sec);



    // uses left and right to change the size of the delay
    if (buttons & BUTTON_LEFT) { 
      l_pressed = true;   
    } else if ( l_pressed & D > 500) {
      D = D - 1000;
      sec = ((D+500) / 1000);
      Serial.println("Seconds:");
      Serial.println(sec);

      l_pressed = false;
    }

    if (buttons & BUTTON_RIGHT) { 
      r_pressed = true;   
    } else if (r_pressed) {
      D = D + 1000;
      sec = ((D+500) / 1000);
      
      Serial.println("Seconds:");
      Serial.println(sec);



      r_pressed = false;
    }

    if (buttons & BUTTON_DOWN) { 
      d_pressed = true;   
    } else if (d_pressed) {
      d_pressed = false;
      shown = false;
      loop();
    }
  }
}

// method for the number of the length of the sequence N
void change_N() {
  lcd.setCursor(0, 0);

  if (shown == false) {
    Serial.println("LENGTH OF SEQUENCE");
    lcd.print("LENGTH OF");
    lcd.setCursor(0, 1);
    lcd.print("SEQUENCE");
    delay(1000);
    lcd.clear();
    shown = true;
  }
  while (true) {
    lcd.setCursor(0, 0);
    int buttons = lcd.readButtons();
    lcd.print(N);

    // use the left and right buttons
    // to change the size of N
    
    if (buttons & BUTTON_LEFT) { 
      l_pressed = true;   
    } else if ( l_pressed & N>2) {
      lcd.clear();
      N= N- 1;
      Serial.println(N);
      l_pressed = false;
    }

    if (buttons & BUTTON_RIGHT) { 
      r_pressed = true;   
    } else if (r_pressed) {
      lcd.clear();
      N= N+ 1;
      Serial.println(N);
      r_pressed = false;
    }


    if (buttons & BUTTON_DOWN) { 
      d_pressed = true;   
    } else if (d_pressed) {

      // allocates memory and wries to the array 
      // containg the correct sequence 
      // and the sequence the user enters.
      S = (int*)malloc(N * sizeof(int));
      A = (int*)malloc(N * sizeof(int));
      for (int i = 0; i <= (N-1) ; i++) {
        S[i] = M[random(0,M_size)];
      }
      Serial.println("Done");
      d_pressed = false;
      s_pressed = false;
      change = true;
      shown = false;
      game_loop();
    }


  }
}




// main game loop
void loop() {
  lcd.setCursor(0, 0);
  while (true) {
    int buttons = lcd.readButtons();
    lcd.setBacklight(-1);

    // handles the main menu
    if (buttons & BUTTON_DOWN) {
      d_pressed = true;
      story = true;
    } else if (d_pressed) {
      d_pressed = false;
      difficulty();
    }

    // uses text to prompt the user to use left and right
    // to change the valuees for practice mode
    if (buttons & BUTTON_SELECT) {
      s_pressed = true;
    } else if (s_pressed) {
      s_pressed = false;
      lcd.clear();
      Serial.println("LEFT AND RIGHT TO CHANGE VALUES");
      lcd.print("LEFT AND RIGHT");
      lcd.setCursor(0,1);
      lcd.print("TO CHANGE VALUES:");
      delay(2000);
      lcd.clear();
      change_M(); // state transition into changing the values
    }

    // state transition to the method for the length of N
    if (change) {
      change_N();
    }
  }
}
