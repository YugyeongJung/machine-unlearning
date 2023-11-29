import * as React from 'react';
import { IconButton, Stack, Button } from '@mui/material';
import eraser from '../icons/eraser.png'
import './style.css'
import { useState, useEffect } from 'react'
import Info from './Info'
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import { useNavigate } from "react-router-dom";
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';



export default function ModelCheck(){
    const [popup, setPopup] = useState(false)
    const [result, setResult] = useState(null);
    const navigate = useNavigate();
    const InfoPopupHandler = () => {
        setPopup(true)
    }
    const delay = ms => new Promise(res => setTimeout(res, ms));

    
    useEffect(() => {
        const ModelCheck = async () => {
            // const res = await fetch("http://localhost:5000/ModelCheck");
            // const data = await res.json();
            // setResult(data)
            // console.log(data)
            await delay(5000);
            setResult('tempdata')
        }
        ModelCheck();
        
    })

    return (
        <div>
            <Stack direction="row">
                <IconButton aria-label="delete" 
                            sx = {{marginRight: 30, marginTop: -3, color: 'black', justifyContent: 'flex-end'}}
                            onClick = {() => InfoPopupHandler()}>
                    <img  src={eraser} style = {{marginLeft: 30, marginTop: 30}}/>
                </IconButton>
            </Stack>
            <div>
                <h1 style  = {{fontSize: 60, fontFamily: "Roboto", color: "#247AFC",  textAlign: "left", marginLeft: 120, marginTop: -10}}>
                    Checking Model
                </h1>
                    { (! result)
                    ?
                    <Box sx={{ display: 'flex', justifyContent: "center", marginTop: 10}}>
                        <CircularProgress size="8rem"/>
                    </Box>
                    :
                    <Button component="label" variant="contained"
                        endIcon={<ArrowForwardIcon style={{marginLeft: 300, fontSize: 50}}/>}
                        sx = {{width: 600, height: 80, borderRadius: "50px", justifyContent: "flex-start", 
                            fontSize: "1.5rem", fontFamily: "Roboto", textTransform: 'none', fontWeight: "200px", marginTop: 10}}
                        onClick = {() => navigate("/TrainingResult")}
                        >
                        <h3 style = {{marginLeft: 15}}>
                            Check Result
                        </h3>
                    </Button>
                    }
                    
                <h3 style  = {{fontSize: 30, fontFamily: "Roboto", fontWeight: 10}}> 
                Checking if your image has been trained... 
                </h3>
                <div style = {{display: 'flex', justifyContent: "center", marginTop: -10}}>
                    <Box component="section" sx={{ p: 1, border: '1px dashed grey', width: 500, fontSize: 25}}>
                        Check Method: Membership Inference
                    </Box>
                </div>
                <hr style = {{marginTop: 150, width: 500, marginLeft: 50}}/>
                <div>
                    <h1 style = {{fontSize: 30, fontFamily: "Roboto", textAlign: "left", marginLeft: 50}}> What is membership inference? </h1>
                    <p style = {{fontSize: 20, fontFamily: "Roboto", textAlign: "left", marginLeft: 50, fontWeight: 10, width: 1000}}>This method works on the idea that an AI model usually does better on data it has been trained on than on new, untrained data. In simpler terms, if your data has been used for training, this method can identify it by detecting a high score in the model's performance. </p>
                </div>
            </div>
            {(popup) && <Info setPopup = {setPopup}/>}
        </div>
    )
}
