import './App.css'
import { useEffect, useState } from 'react'
import { useDispatch } from 'react-redux'
import { login, logout } from './store/authSlice';
import { Footer, Header } from './components';
import { Outlet } from 'react-router-dom';

function App() {

  return (
    <>
      <div className='min-h-screen flex flex-wrap content-between bg-gray-500'>
        <div className='w-full block'>
          <Header />
          <main>
            <Outlet />
          </main>
          <Footer />
        </div>
      </div>
    </>
  )
}

export default App
