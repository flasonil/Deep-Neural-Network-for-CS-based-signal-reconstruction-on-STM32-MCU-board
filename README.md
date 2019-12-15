<h1>Lab of Statistical Signal Processing - Deep Neural Network for CS based signal reconstruction on STM32 MCU board</h1>

<h2>Project Requirement</h2>
<ul>
    <li>System Workbench for STM32 (or equivalent IDE)</li>
    <li>STCube MX v5.2.0</li>
    <li>STCube MX AI package v4.1.0</li>
    <li>Python v3.7</li>
    <li>Tensorflow v2.0</li>
    <li>Jupyter Notebook</li>
    <li>SMT32 Board - NucleoH743ZI2 and STM32F4DISCO have been used for this demo</li>
</ul>

<h2>Compressed Sensing Basics</h2>
<p>CS hinges on the assumption that
x is <a href="https://www.codecogs.com/eqnedit.php?latex=$\kappa$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\kappa$" title="$\kappa$" /></a>-sparse, i.e., in the simplest possible setting, that an
orthonormal matrix <a href="https://www.codecogs.com/eqnedit.php?latex=$S$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$S$" title="$S$" /></a> exists (whose columns are the vectors
of the sparsity basis) such that when we express <a href="https://www.codecogs.com/eqnedit.php?latex=$x=S\xi$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$x=S\xi$" title="$x=S\xi$" /></a>, then
the vector <a href="https://www.codecogs.com/eqnedit.php?latex=$\mathrm{\xi&space;=&space;(\xi_{1},...,\xi_{n-1})}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathrm{\xi&space;=&space;(\xi_{1},...,\xi_{n-1})}$" title="$\mathrm{\xi = (\xi_{1},...,\xi_{n-1})}$" /></a> does not contain more than <a href="https://www.codecogs.com/eqnedit.php?latex=$\kappa&space;<&space;n$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\kappa&space;<&space;n$" title="$\kappa < n$" /></a> non-zero entries.
The fact that <a href="https://www.codecogs.com/eqnedit.php?latex=$x$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$x$" title="$x$" /></a> depends only on a number of scalars that
is less than its sheer dimensionality hints at the possibility
of compressing it. CS does this by applying a linear operator
<a href="https://www.codecogs.com/eqnedit.php?latex=$\mathcal{L_{\mathrm{A}}}:\mathbb{R^\mathrm{m}}\mapsto\mathbb{R^\mathrm{n}}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\mathcal{L_{\mathrm{A}}}:\mathbb{R^\mathrm{m}}\mapsto\mathbb{R^\mathrm{n}}$" title="$\mathcal{L_{\mathrm{A}}}:\mathbb{R^\mathrm{m}}\mapsto\mathbb{R^\mathrm{n}}$" /></a>  depending on the acquisition (or encoding)
matrix <a href="https://www.codecogs.com/eqnedit.php?latex==$A\in\mathbb{R^\mathrm{m\times&space;n}}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?=$A\in\mathbb{R^\mathrm{m\times&space;n}}$" title="=$A\in\mathbb{R^\mathrm{m\times n}}$" /></a> with m < n and defined in such a way that
<a href="https://www.codecogs.com/eqnedit.php?latex=$x\in\mathbb{R^\mathrm{n}}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$x\in\mathbb{R^\mathrm{n}}$" title="$x\in\mathbb{R^\mathrm{n}}$" /></a> can be retrieved from <a href="https://www.codecogs.com/eqnedit.php?latex=$y=\mathcal{L_{\mathrm{A}}(x)}\in\mathbb{R^\mathrm{m}}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$y=\mathcal{L_{\mathrm{A}}(x)}\in\mathbb{R^\mathrm{m}}$" title="$y=\mathcal{L_{\mathrm{A}}(x)}\in\mathbb{R^\mathrm{m}}$" /></a>. The ratio
<a href="https://www.codecogs.com/eqnedit.php?latex=$n/m$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$n/m$" title="$n/m$" /></a> is the compression ratio and will be indicated by CR.
<cite>[Mangia, Mauro & Prono, Luciano & Pareschi, Fabio & Rovatti, Riccardo & Setti, Gianluca. (2019). Deep Neural Oracles for Short-window Optimized Compressed Sensing of Biosignals. 10.36227/techrxiv.10049843.v2. ]</cite></p>

<p>More specifically, in this demo a serial communication has been established between the MCU and Python enironment. A master computer feeds y_test data to the MCU which in turn computes the output <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{s}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{s}" title="\hat{s}" /></a> from the neural network and sends back such data to the computer. Finally, RSNR and the reconstructed input <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{x}" title="\hat{x}" /></a> are computed by the Jupyter Notebook.</p>

![Figure 1](Cattura.png "Figure 1")
    
    
<h2>MCU Preparation</h2>
<p>After having installed STCube MX version 5.2 and AI extension version 4.1 on a virtual machine with Windows 7 (Neither "Validation on Desktop" nor "Validation on Target" tasks appear to work on Windows 10), I have followed these steps:
<ol>
    <li>Click on "Access Board Selector", choose the appropriate device and click on "Start Project".</li>
    <li>Enable UART peripheral: USART3 peripheral of NUCLEO-144 board is directly connected to the ST-LINK therefore just a USB cable wil be needed for both programming and communication. Select USART2 from "Connectivity" dropdown menu and select "Asynchronous".</li>
    <li>Cick on "Additional Software" -> "STMicroelectronics.X_CUBE_AI" -> "Validation" (or "Application Template" to generate the code)</li>
    <li>Open Artificial Intelligence menu from the left-hand side, select the UART peripheral under "Platform Setting" previously enabled and click on "Add Network". Choose "Keras", "Saved Model" and upload the h5 format file containing the trained Neural Network.</li>
    <li>The model can be accurately examined with the options "Analyze", "Validate on Desktop", "Validate on target". Pay attention to select the correct COM port associated to the ST-LINK. This can be checked from "Control Panel" in Windows.</li>
    <li>Choose the adopted IDE (for this demo SystemWorkbench for STM32) in "Project Manager" options and run "Generate Code" (Remember to switch mode from Validation to Application Template, see point 3)</li>
</ol>
The AI project with the Neural Network can now be moved to the actual OS and the Virtual Machine be turned off. Some modification of the code are needed in order for the serial communication to work properly.
<ul>
    <li>Initialize the UART peripheral by calling this procedure inside main.c</li>
        
```c
MX_USART3_UART_Init();
```
   <li>Define globally the following variables inside app_x-cube-ai.c</li>
   
```c
UART_HandleTypeDef huart3;
ai_float outdata[AI_NETWORK_OUT_1_SIZE];
uint8_t shat[AI_NETWORK_OUT_1_SIZE];
ai_float threshold = 0.0352120689269106; //O_min threshold
```

   <li>Modify MX_X_CUBE_AI_Process() function</li>

```c
void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 1 */
	int nb_run = 100;
    AI_ALIGNED(4)
	//INPUT AND OUTPUT BUFFERS
    static ai_i8 out_data[AI_NETWORK_OUT_1_SIZE_BYTES];
    static ai_u8 in_data[AI_NETWORK_IN_1_SIZE_BYTES];

    /* Perform nb_rub inferences (batch = 1) */
    while (--nb_run) {
        /* ---------------------------------------- */
        /* Data generation and Pre-Process          */
        /* ---------------------------------------- */
    	//FILL THE INPUT BUFFER
    	while(HAL_UART_Receive(&huart3, in_data, sizeof(in_data), HAL_MAX_DELAY)!=HAL_OK);
        /* Perform the inference */
        aiRun(in_data, out_data);
        for (ai_size i=0;  i < AI_NETWORK_OUT_1_SIZE; i++ ){
        	//CASTING TO AI_FLOAT THE OUTPUT OF THE NN
        	outdata[i] = ((ai_float *)out_data)[i];
        	//COMPUTING S_HAT
        	if(outdata[i] > threshold) shat[i] = 0x01;
        	else shat[i] = 0x00;
        }
        //TRANSMIT S_HAT
        while(HAL_UART_Transmit(&huart3,shat,sizeof(shat),0xFFFF)!=HAL_OK);
    }
    /* USER CODE END 1 */
}
```   
</ul>
The MCU is now ready to communicate with the computer. If the communication does not work make sure if the baud rate specified in the COM port setting in the control panel is consistent with the value present in UART_Init() procedure.
</p>

<h2>Script Python</h2>
<p>
The following script essentially imports the dataset, establishes a serial connection with the MCU and computes support mismatch and Recontruction SNR with the $\hat{s}$ vectors provided by the microcontroller.
</p>
